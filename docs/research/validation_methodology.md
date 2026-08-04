# Metodología de validación estadística

**Ámbito.** Contrato metodológico para el módulo `earnings_alpha.stats` (`docs/ARCHITECTURE.md` §3.8)
y para todo agente que reporte el rendimiento de un factor (ángulo A), de un estudio de evento
(ángulo B) o de un backtest. Define **qué se calcula, con qué fórmula exacta, y qué umbral hay que
superar** para que una señal se considere viva en este repositorio.

La regla de oro de `ARCHITECTURE.md` §3.8 es: *toda métrica de rendimiento debe reportar su intervalo
de confianza; un Sharpe sin banda de error no se acepta.* Este documento es la especificación de esa
regla.

**Fecha de compilación:** 2026-08-04.

---

## 0. Nota metodológica sobre las fuentes

`WebFetch` devuelve **HTTP 403 para todos los dominios** en este contenedor (verificado contra
`davidhbailey.com`, `portfoliooptimizer.io`, `en.wikipedia.org`). La revisión se ha construido con
búsquedas web cuyos resultados sintetizan resúmenes, más reconstrucción analítica.

Consecuencia práctica, y hay que asumirla con honestidad:

- Toda fórmula de este documento ha sido **verificada numéricamente** (Monte Carlo o comprobación de
  consistencia contra una identidad conocida). Los resultados de esas verificaciones están **en el
  propio texto**, sección por sección, y son reproducibles.
- Las cifras que provienen de un abstract y **no** he podido contrastar van marcadas con ⚠.

**Cuatro hallazgos propios de esta revisión**, todos con consecuencia directa de implementación:

1. **§5.4** — La transcripción del `E[max SR]` de Bailey–López de Prado que más circula por la web
   escribe el segundo término como `Z⁻¹[1 − 1/(N·e⁻¹)]`. **Es un error de signo en el exponente**: la
   forma correcta es `N·e`. Con `N=1000` la versión correcta da 3,2551 (Monte Carlo: 3,2422) y la
   errónea 2,9111, un **11 % de subestimación** que se traduce directamente en aceptar estrategias
   falsas.
2. **§5.2** — En el PSR, `γ̂₄` es la **curtosis cruda** (3 para la normal), no el exceso.
   `scipy.stats.kurtosis` devuelve el **exceso** por defecto (`fisher=True`). Restar 3 de más rompe la
   identidad con el error estándar de Lo (2002), que sirve aquí de test de consistencia exacto.
3. **§3.2 y §3.3** — El t-estadístico ingenuo de una serie de IC construida con retornos *forward* de
   `h` días muestreada a diario está inflado por un factor **√h** (medido: ×4,64 con `h=21`,
   ×7,98 con `h=63`; teoría √h = 4,58 y 7,94). Newey–West con `L ≥ h−1` lo reduce a ×1,25, pero esa
   inflación residual **no desaparece: se estanca**, dejando una tasa de rechazo del 11,7 % frente al
   5 % nominal para todo `h ≥ 10`. El muestreo disjunto sí es exacto (5,2 %).
4. **§3.5** — El solapamiento **transversal** (todas las empresas del S&P 500 presentando resultados
   en las mismas seis semanas) es cuantitativamente **más grave** que el temporal. Con 473 eventos por
   trimestre y una correlación media entre residuos de solo `ρ̄ = 0,05`, el t-estadístico se infla
   **×5,09** (verificado por Monte Carlo: ×5,081) y los 473 eventos valen lo que **19 independientes**.

---

## 1. El problema concreto de este repositorio

Antes de las fórmulas, la estructura de dependencia de los datos, porque **todo lo demás se deriva de
ella**. Aplicar un test estándar a este panel sin entender su dependencia produce t-estadísticos que
se equivocan por factores de 4 a 8, no por decimales.

### 1.1 Aritmética real del panel

Medido sobre `data/seed/sp500_historical_components.csv`:

| Magnitud | Valor |
|---|---|
| Snapshots diarios de composición | 3.482 |
| Rango | 1996-01-02 → 2025-08-23 (29,6 años) |
| Tamaño medio del universo | 472,6 (mín. 442, máx. 505) |
| Tickers distintos que han pasado por el índice | 1.128 |
| Eventos de resultados aproximados | ≈ 56.000 (4 × 473 × 29,6) |
| Trimestres disponibles | 119 |

Parece muchísimo. **No lo es**, y esa es la tesis de este documento.

### 1.2 Cuatro fuentes de dependencia, en orden de gravedad

**(a) Solapamiento transversal por agrupamiento de fechas (el peor).**
Las ~473 empresas del índice presentan resultados concentradas en unas seis semanas por trimestre.
Sus retornos anormales comparten factores comunes (mercado, sector, macro del día). Si los residuos
de los eventos de un mismo trimestre tienen correlación media `ρ̄`, la varianza del CAR medio no es
`σ²/n` sino:

```
Var(CAR medio) = (σ²/n) · [1 + (n−1)·ρ̄]
```

y por tanto el número efectivo de observaciones independientes es:

```
n_eff = n / [1 + (n−1)·ρ̄]
```

Medido (n = 473 eventos por trimestre, 119 trimestres):

| `ρ̄` | `n_eff` por trimestre | inflación del t | obs. efectivas totales |
|---|---|---|---|
| 0,02 | 45 (de 473) | ×3,3 | 5.372 |
| 0,05 | 19 (de 473) | ×5,1 | 2.280 |
| 0,10 | 10 (de 473) | ×7,1 | 1.163 |

**Verificación Monte Carlo** (n=500, `ρ̄`=0,05, 20.000 réplicas): `sd(t ingenuo) = 5,081` frente al
5,094 teórico. La fórmula es exacta.

Es decir: los "56.000 eventos" del repo son, en el mejor de los casos, del orden de **2.000–5.000
observaciones efectivas**. Este es *el* número que debe gobernar cualquier afirmación de significancia
en el ángulo B. Es exactamente el resultado de Kolari–Pynnönen (2010): con agrupamiento de fechas,
incluso una correlación cruzada baja produce sobre-rechazo severo.

**(b) Solapamiento temporal por retornos *forward*.**
Si `IC_t` se calcula con el retorno de `h` días vista y la serie se muestrea a diario, `IC_t` e
`IC_{t+1}` comparten `h−1` días de retorno. La serie hereda una estructura MA(`h−1`) con
`ρ_k ≈ (h−k)/h`, y el ratio de varianza es exactamente `h`. Ver §3.2 para la medición.

**(c) Persistencia de los propios factores.**
Los fundamentales cambian trimestralmente. Un factor tipo Piotroski o *accruals* es casi constante
entre presentaciones: la señal está autocorrelacionada con `ρ` cercano a 1 a horizonte diario. Esto
afecta a los retornos de la cartera, no solo a la IC.

**(d) Solapamiento de ventanas de estimación/evento.**
`abnormal_returns` usa una ventana de estimación `(-250, -40)`. Dos eventos consecutivos de la *misma
empresa* distan ~63 sesiones: la ventana de estimación del trimestre `q+1` **contiene** la ventana de
evento del trimestre `q`. Esto no es look-ahead pero sí contamina la estimación del modelo de mercado
con el propio efecto que se quiere medir.

### 1.3 Consecuencia de diseño

> **Ninguna cifra de significancia de este repo puede calcularse sin declarar explícitamente qué
> corrección de dependencia se ha aplicado.** Un `t` sin etiqueta de corrección es un `t` inválido y
> el código debe negarse a emitirlo.

Propuesta de contrato para `earnings_alpha.stats` (el agente propietario decide la firma final):

```python
@dataclass(frozen=True)
class SignificanceResult:
    """Resultado de un contraste, con su corrección de dependencia declarada."""
    estimate: float
    std_error: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_eff: float                  # observaciones efectivas tras corregir dependencia
    method: str                   # "newey_west", "kolari_pynnonen", "stationary_bootstrap"...
    nw_lags: int | None = None
    notes: str = ""
```

---

## 2. Coeficiente de información (IC) y rank IC

### 2.1 Definición exacta

Sea `s_{i,t}` la señal del activo `i` **conocible en `t`** (ya desplazada, ver `pit.asof_join`), y
`r_{i,t→t+h}` su retorno *forward* de `h` sesiones. Para cada fecha `t`, sobre el universo
`U_t` de `n_t` activos que están en el índice en `t`:

**IC de Pearson**

```
             Σ_{i∈U_t} (s_{i,t} − s̄_t)(r_{i,t→t+h} − r̄_t)
IC_t = ─────────────────────────────────────────────────────────────
        sqrt[ Σ (s_{i,t} − s̄_t)² ] · sqrt[ Σ (r_{i,t→t+h} − r̄_t)² ]
```

**Rank IC (Spearman)** — idéntica fórmula sustituyendo `s` y `r` por sus rangos dentro de la sección
cruzada de la fecha `t`.

**Cuál usar en este repo: rank IC, siempre, como métrica principal.** Razones concretas:

1. Los factores fundamentales tienen colas patológicas. Un *earnings yield* con beneficio cercano a
   cero, un *accrual* con activos pequeños o un FCF yield negativo generan valores de |z| > 20. La IC
   de Pearson sobre esos datos mide el efecto de tres observaciones.
2. El retorno tiene curtosis alta, y en la ventana de evento muchísimo más (gaps de ±20 %).
3. La IC de Pearson no es invariante a las transformaciones monótonas que aplica `signals`
   (`winsorize`, `rank_pct`), por lo que la métrica cambiaría según el preprocesado. La rank IC no.

La IC de Pearson se reporta **como diagnóstico secundario**: si `IC_pearson >> IC_rank`, el factor
vive en las colas y su implementabilidad es dudosa (concentración, capacidad, coste).

### 2.2 Convenciones obligatorias del repo

- **Universo**: `U_t` es la pertenencia PIT al índice en `t` (`universe.members_on(t)`), nunca la
  lista actual. Sin esto la IC está sesgada al alza por supervivencia.
- **Mínimo de sección cruzada**: si `n_t < 30`, `IC_t` es `NaN`, no un número. Con `n_t` pequeño el
  estimador de correlación tiene un sesgo y una varianza que rompen todo lo posterior.
- **Neutralización**: reportar `IC` cruda y `IC` neutralizada por sector (`signals.neutralize`).
  Una IC que desaparece al neutralizar por sector es una apuesta sectorial disfrazada.
- **Signo**: por `ARCHITECTURE.md` §3.4, mayor valor = más alcista, luego una IC positiva es lo
  esperado. Una IC negativa significativa es un hallazgo, no un error de signo que se "arregla"
  invirtiendo la señal *a posteriori* — invertir tras ver el resultado es una prueba más que hay que
  contar en `N` (§6).

### 2.3 Error estándar de una IC de un solo día

Para una sola fecha, con `n` activos, la transformación de Fisher normaliza la distribución:

```
z_t = arctanh(IC_t) = ½ · ln[(1 + IC_t)/(1 − IC_t)]
```

- **Pearson**: `SE(z) = 1/√(n−3)`, exacto asintóticamente bajo normalidad bivariante.
- **Spearman**: `SE(z) = √[(1 + ρ̂²/2)/(n−3)]` (Bonett–Wright, 2000). Como aproximación práctica en
  el rango relevante (`|ρ| < 0,1`), `SE(z) ≈ 1,03/√(n−3)` (Fieller–Hartley–Pearson).

Intervalo de confianza: `tanh(z ± z_{α/2}·SE(z))`.

**Magnitud medida** — qué IC haría falta para que **un solo día** fuese significativo:

| `n` | `SE(z)` Pearson | `SE(z)` Spearman | \|IC\| mínima para \|t\|>2 en un día |
|---|---|---|---|
| 50 | 0,1459 | 0,1502 | 0,284 |
| 100 | 0,1015 | 0,1045 | 0,200 |
| 250 | 0,0636 | 0,0655 | 0,127 |
| 500 | 0,0449 | 0,0462 | 0,089 |

Con `n = 475` (el universo típico de este repo) una IC diaria individual necesitaría ser ≈ 0,09 para
ser significativa. **Las IC reales de factores viables están en 0,01–0,05.** Corolario operativo: *la
IC de un día no significa nada nunca*. Toda la evidencia está en la **serie temporal** de IC, y ahí es
donde hay que ser cuidadoso.

### 2.4 Serie temporal de IC: media, IC-IR y t-estadístico

Sobre `T` fechas:

```
ĪC   = (1/T) Σ_t IC_t
σ_IC = sd muestral de {IC_t}
IC_IR = ĪC / σ_IC                       (por periodo)
IC_IR_anual = IC_IR · √(periodos por año)
```

**t-estadístico ingenuo** (el que hay que evitar):

```
t_naive = ĪC / (σ_IC/√T) = IC_IR · √T
```

Este `t_naive` es válido **solo si** las `IC_t` son independientes, lo cual sucede únicamente cuando
`h = 1` **y** el muestreo es diario, o cuando el muestreo es disjunto (una observación cada `h`). En
cualquier otro caso está inflado, y la §3 explica cuánto.

---

## 3. Newey–West: error estándar y selección de lags

### 3.1 Fórmula exacta

Sea `x_t` la serie cuyo promedio queremos contrastar (`IC_t`, el retorno diario del spread `Q5−Q1`, el
retorno de una cartera). Con `d_t = x_t − x̄`:

**Autocovarianzas muestrales**

```
γ̂_k = (1/T) · Σ_{t=k+1}^{T} d_t · d_{t−k}          k = 0, 1, …, L
```

**Varianza de largo plazo con kernel de Bartlett**

```
Ŝ_NW = γ̂_0 + 2 · Σ_{k=1}^{L} w_k · γ̂_k      con     w_k = 1 − k/(L+1)
```

**Error estándar y t-estadístico**

```
SE_NW(x̄) = √(Ŝ_NW / T)
t_NW      = x̄ / SE_NW(x̄)
```

Los pesos de Bartlett `w_k = 1 − k/(L+1)` no son cosméticos: garantizan que `Ŝ_NW ≥ 0`, cosa que la
suma sin ponderar (Hansen–Hodrick) no garantiza y que en la práctica produce varianzas negativas con
`L` grande y `T` moderado. **En este repo se usa siempre el kernel de Bartlett.**

Para regresión (Fama–MacBeth, regresiones predictivas), la misma construcción sobre
`d_t = x_t·û_t` da la matriz HAC `V = (X'X)⁻¹ · Ŝ_NW · (X'X)⁻¹`.

### 3.2 Selección de lags: la regla del repo

Las dos reglas estándar:

| Regla | Fórmula | Origen |
|---|---|---|
| Newey–West / Stock–Watson | `L = ⌊4·(T/100)^(2/9)⌋` | práctica dominante, kernel de Bartlett |
| Newey–West (1994) automática | `L = ⌊4·(T/100)^(2/25)⌋` como semilla, refinada con la regla *data-driven* | Newey & West (1994), *RES* |

Valores calculados:

| `T` | `⌊4(T/100)^(2/9)⌋` | `⌊4(T/100)^(2/25)⌋` |
|---|---|---|
| 252 (1 año diario) | 4 | 4 |
| 504 | 5 | 4 |
| 1.260 (5 años diario) | 7 | 4 |
| 2.520 | 8 | 5 |
| 5.040 | 9 | 5 |
| 7.500 (30 años diario) | 10 | 5 |

**El problema: ninguna de las dos reglas conoce `h`.** Con 30 años de datos diarios la regla
automática pide `L = 10`, pero si la IC usa retornos a 63 días la MA inducida tiene orden 62. La regla
automática se queda corta por un factor 6.

> **Regla vinculante del repo:**
> ```
> L = max( ⌊4·(T/100)^(2/9)⌋ , h − 1 )
> ```
> donde `h` es el horizonte del retorno *forward* en periodos de muestreo. El término `h−1` es un
> **suelo**, no una alternativa: cubre la MA determinista inducida por el solapamiento; el término
> automático cubre la autocorrelación económica residual.

### 3.3 Medición: cuánto corrige realmente Newey–West

Simulación con series MA(`h−1`) de media cero verdadera, `T = 2.500`, 2.000 réplicas. Bajo la
hipótesis nula, `sd(t)` debería ser 1,000 y la tasa de rechazo al 5 % debería ser 0,050.

| `h` | `L_NW` | `T/h` | `sd(t)` ingenuo | rechazo | `sd(t)` NW | rechazo | `sd(t)` disjunto | rechazo |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 2.500 | 1,010 | 0,052 | 1,012 | 0,054 | 1,010 | 0,052 |
| 5 | 8 | 500 | 2,210 | **0,382** | 1,093 | 0,073 | 0,989 | 0,049 |
| 10 | 9 | 250 | 3,089 | **0,521** | 1,197 | 0,102 | 0,978 | 0,045 |
| 21 | 20 | 119 | 4,639 | **0,679** | 1,247 | 0,117 | 1,011 | 0,054 |
| 42 | 41 | 59 | 6,567 | **0,764** | 1,255 | 0,115 | 1,013 | 0,054 |
| 63 | 62 | 39 | 7,976 | **0,808** | 1,254 | 0,117 | 1,003 | 0,052 |

**Tres lecturas, todas accionables:**

1. **El t ingenuo con solapamiento es catastrófico.** Con `h = 21` (el horizonte natural para un
   factor fundamental), la tasa de rechazo real al nominal 5 % es del **67,9 %**. Un factor de puro
   ruido "sale significativo" dos de cada tres veces. Con `h = 63`, el **80,8 %**.
2. **Newey–West con `L ≥ h−1` corrige el grueso, pero deja un residuo que no desaparece.** La
   inflación cae de ×7,98 a ×1,25, pero **se estanca ahí**: la tasa de rechazo se estabiliza en
   ≈ 11,7 % (2,3× el nominal) para todo `h ≥ 10`. No es que Newey–West falle más con `h` grande — es
   que su sesgo residual es una constante, no una función de `h`. Un `t_NW` de 2,0 sobre datos
   solapados corresponde en realidad a un `t` efectivo de ≈ 1,6.
3. **El muestreo disjunto es exacto en todo el rango** (rechazo 4,5–5,4 %), al precio de reducir la
   muestra a `T/h` observaciones.

> **Regla vinculante:**
> - Con `h ≤ 5`: Newey–West con `L = max(⌊4(T/100)^(2/9)⌋, h−1)` es suficiente.
> - Con `h > 5`: se usa Newey–West **y** se reporta el `t` con muestreo disjunto. En caso de
>   discrepancia manda el disjunto.
> - En cualquier caso, un `t_NW` sobre datos solapados se compara contra un umbral **inflado un
>   25 %** respecto al nominal (es decir, exigir `t_NW ≥ 3,0` equivale a exigir `t ≈ 2,4` real).
>   Esta es una de las razones por las que los umbrales de §14 son altos.

### 3.4 Newey–West no arregla el solapamiento transversal

Punto crítico y fácil de pasar por alto: Newey–West corrige la dependencia **a lo largo del tiempo**
de una serie ya agregada. **No corrige** la correlación **dentro** de la sección cruzada de una fecha.

En la serie de IC eso está automáticamente contemplado, porque `IC_t` es un escalar por fecha: la
correlación transversal entre activos afecta a la *varianza* de `IC_t`, y `σ_IC` la recoge
empíricamente. **Por eso la serie de IC es un estimador robusto** y es la métrica principal del
ángulo A.

En un estudio de evento **no**: si se apilan 56.000 eventos en una regresión *pooled* y se aplica
Newey–West sobre el tiempo-evento, la correlación entre eventos simultáneos queda sin corregir. Ahí
hace falta §4.

---

## 4. Estudios de evento con solapamiento: el ángulo B

### 4.1 Por qué el t transversal clásico falla aquí

El test clásico de estudio de evento (Brown–Warner, Patell) asume que los residuos de los distintos
eventos son independientes. Eso vale para eventos idiosincrásicos dispersos en el tiempo (fusiones,
demandas). **No vale para resultados trimestrales**, donde 473 empresas presentan en la misma ventana
de seis semanas y sus residuos comparten el factor de mercado de cada día.

La magnitud está en §1.2(a): inflación ×5,1 con `ρ̄ = 0,05`.

### 4.2 Correcciones por orden de preferencia

**(1) Cartera calendario (la más robusta y la que este repo prefiere).**
En vez de promediar eventos, se construye una cartera diaria con todas las empresas que están dentro
de su ventana de evento en esa fecha, y se contrasta la **serie temporal** del retorno de esa cartera
con Newey–West. La correlación transversal desaparece por construcción: cada día es una sola
observación. Coste: se pierde la estructura de tiempo-evento (no ves el perfil por `tau`).

Es el mismo argumento que hace de la serie de IC un buen estimador (§3.4).

**(2) Boehmer–Musumeci–Poulsen (BMP) ajustado por Kolari–Pynnönen.**
BMP (1991) estandariza cada retorno anormal por su desviación típica de la ventana de estimación,
lo que corrige la varianza inducida por el propio evento. Kolari–Pynnönen (2010) añaden la corrección
por correlación cruzada:

```
                        mean(SCAR)
t_KP = -------------------------------------------------
        ( sd(SCAR) / sqrt(n) ) * sqrt( 1 + (n-1)*rho_bar )
```

donde `SCAR_i` es el CAR estandarizado del evento `i` y `ρ̄` es la correlación media entre los
residuos de la ventana de estimación de todos los pares de eventos. **`ρ̄` se estima de los datos**,
sobre la ventana de estimación (que por construcción no contiene el evento).

Kolari–Pape–Pynnönen (2018) extienden el ajuste al caso de **ventanas parcialmente solapadas**, que es
exactamente el caso de este repo: dos empresas que presentan con tres días de diferencia tienen
ventanas `[-30,+60]` solapadas en un 93 %. El ajuste sustituye `ρ̄` por `ρ̄ · λ̄`, donde `λ̄` es el
porcentaje medio de solapamiento entre pares de ventanas. ⚠ La forma funcional exacta de este último
ajuste no la he podido contrastar contra el texto completo; **verificar antes de usar** (anotado en
`docs/OPEN_QUESTIONS.md`).

**(3) Bootstrap por bloques de fechas.**
Remuestrear **fechas de calendario completas** (con todos los eventos de esa fecha juntos) en vez de
eventos individuales preserva la correlación transversal automáticamente. Es la opción preferida
cuando `ρ̄` es difícil de estimar. Combinar con el bootstrap estacionario de §8 si además hay
dependencia temporal entre fechas.

**(4) Errores estándar agrupados por fecha (*clustered by date*).**
En una regresión *pooled* de `CAR_i` sobre características, agrupar por fecha de evento. Petersen
(2009) documenta que en paneles financieros el agrupamiento correcto cambia los errores estándar por
factores grandes; con agrupamiento por fecha, el número de clusters es el número de fechas de evento
distintas (aquí, ~2.500 en 30 años), suficiente para la asintótica.
Si además hay persistencia entre fechas, Driscoll–Kraay (robusto a dependencia transversal *y*
serial) es la opción conservadora.

### 4.3 Solapamiento entre ventana de estimación y evento anterior

Con `estimation = (-250, -40)` y eventos cada ~63 sesiones, la ventana de estimación del trimestre
`q+1` contiene las ventanas de evento de los trimestres `q`, `q−1` y `q−2`. Esto sesga al alza la
varianza estimada (`σ` de la ventana de estimación incluye tres saltos de resultados), lo que hace el
test **conservador** en el numerador estandarizado pero contamina la estimación de beta.

Mitigación recomendada: **excluir de la ventana de estimación las sesiones `[-2, +2]` alrededor de
cualquier evento de resultados previo de la misma empresa**. Es barato de implementar (los eventos ya
están en el panel) y elimina el sesgo sin perder apenas observaciones (12 de 210 sesiones).

### 4.4 Solapamiento entre ventanas de evento consecutivas del mismo emisor

Con `post = 60` sesiones y eventos cada ~63, las ventanas de evento consecutivas de una empresa
**casi** se tocan. Con `post > 63` se solapan y el mismo día de retorno cuenta dos veces con
etiquetas distintas. Consecuencias:

- Para el cálculo de CAR: usar `post ≤ 60` evita el problema. Si se necesita `post > 63`, hay que
  truncar la ventana en la siguiente presentación.
- Para la CV purgada (§7): el *span* de información de un evento es
  `[T − pre − lookback_features , T + post]`. Con `pre=30`, `post=60` y un `lookback` de 250 sesiones
  para el modelo de mercado, el span es de **340 sesiones**. Ese es el `h` que hay que purgar, no 60.
  Este número gobierna el coste de la CV purgada y es donde más gente se equivoca.

---

## 5. Sharpe: probabilístico, deflactado y longitud mínima de registro

Toda esta sección trabaja con el Sharpe **por observación** (no anualizado):

```
SR̂ = (r̄ − r_f) / σ̂_r        con r̄, σ̂_r calculados a la frecuencia de muestreo
```

La conversión es `SR_anual = SR̂ · √(periodos por año)`, pero **las fórmulas de PSR/DSR usan `SR̂` sin
anualizar**. Meter un Sharpe anualizado en ellas es el error de implementación más común y produce
p-valores absurdos.

### 5.1 Error estándar del Sharpe

Bajo retornos iid normales (Lo, 2002):

```
SE(SR̂) = √[ (1 + SR̂²/2) / T ]
```

Con retornos no normales (Mertens; Christie; Opdyke; Bailey–López de Prado):

```
SE(SR̂) = √[ (1 − γ̂₃·SR̂ + ((γ̂₄ − 1)/4)·SR̂²) / (T − 1) ]
```

donde `γ̂₃` es la asimetría y **`γ̂₄` es la curtosis CRUDA** (3 para la normal).

**Verificación de consistencia (exacta).** Con `γ̂₃ = 0` y `γ̂₄ = 3` el factor debe reducirse al de Lo:

```
SR̂ = 0,10:
  Lo (2002)     √(1 + SR²/2)            = 1,00249688
  Bailey, γ₄=3 (curtosis cruda)         = 1,00249688   ✓ coincide
  Bailey, γ₄=0 (curtosis en exceso)     = 0,99874922   ✗ no coincide
```

> **Trampa de implementación.** `scipy.stats.kurtosis(x)` devuelve la curtosis **en exceso**
> (`fisher=True` por defecto). Hay que usar `scipy.stats.kurtosis(x, fisher=False)` o sumar 3.
> Usar el exceso hace el denominador **más pequeño**, es decir, **infla** la significancia: falla en
> la dirección peligrosa.

**Cuánto importa realmente la no-normalidad** (asimetría −0,5, curtosis cruda 8, típicas de una
estrategia long-short de acciones):

| Frecuencia | `SR` anual | `SR` por obs. | efecto en el SE frente al gaussiano |
|---|---|---|---|
| diaria | 0,5 | 0,0315 | +0,8 % |
| diaria | 1,0 | 0,0630 | +1,8 % |
| diaria | 2,0 | 0,1260 | +4,0 % |
| semanal | 0,5 | 0,0693 | +2,0 % |
| semanal | 1,0 | 0,1387 | +4,5 % |
| semanal | 2,0 | 0,2774 | +10,7 % |
| mensual | 0,5 | 0,1443 | +4,7 % |
| mensual | 1,0 | 0,2887 | +11,3 % |
| mensual | 2,0 | 0,5774 | **+26,7 %** |

**Hallazgo honesto, y tiene una consecuencia de diseño.** El efecto de la no-normalidad escala con
`SR` **por observación**, es decir, con la raíz del periodo de muestreo. A frecuencia **diaria** la
corrección es de un 1–4 %: real, hay que aplicarla, pero **no es lo que salva de las falsas señales**.
A frecuencia **mensual** con Sharpe alto llega al 27 % y sí es material.

Dos corolarios:

1. Si el backtest se evalúa a frecuencia diaria (lo habitual aquí), el PSR aporta poco sobre el
   t-estadístico clásico. **Lo que mata las falsas señales por órdenes de magnitud es la deflación por
   número de pruebas (§5.4).** Implementar el PSR y olvidar el DSR no resuelve nada.
2. Reportar el Sharpe a frecuencia mensual (práctica común en la industria) **hace la corrección más
   necesaria, no menos**. La agregación no elimina las colas: concentra el efecto en el estadístico.

### 5.2 Sharpe probabilístico (PSR)

Bailey y López de Prado (2012, 2014). Es la probabilidad de que el Sharpe verdadero supere un umbral
de referencia `SR*`:

```
                     (SR_hat - SR_star) * sqrt(T - 1)
PSR(SR_star) = Phi( ---------------------------------------------- )
                     sqrt( 1 - g3*SR_hat + ((g4 - 1)/4)*SR_hat^2 )
```

con `g3 = asimetría muestral`, `g4 = curtosis muestral CRUDA` (3 para la normal). En notación del
paper: `γ̂₃ ≡ g3`, `γ̂₄ ≡ g4`.

- `Phi` (`Φ`) = función de distribución acumulada normal estándar.
- `SR̂`, `SR*` en unidades **por observación**.
- `γ̂₄` **curtosis cruda**.
- `PSR(0)` es el complementario del p-valor de una cola de `H₀: SR = 0`.

Interpretación: `PSR(0) = 0,95` ⟺ 95 % de confianza en que el Sharpe verdadero es positivo, ya
corregido por longitud de muestra, asimetría y curtosis. **Pero no por selección.**

### 5.3 Longitud mínima de registro (MinTRL)

Despejando `T` de la ecuación anterior con nivel de confianza `1−α`:

```
MinTRL = 1 + [ 1 − γ̂₃·SR̂ + ((γ̂₄−1)/4)·SR̂² ] · ( Z_α / (SR̂ − SR*) )²
```

Valores calculados (`SR* = 0`, `α = 5 %`, `Z_α = 1,645`, datos diarios):

| `SR` anual | MinTRL (asim. −0,5, curt. 8) | en años | MinTRL (gaussiano) | en años |
|---|---|---|---|---|
| 0,5 | 2.776 obs | **11,02** | 2.730 | 10,83 |
| 1,0 | 709 obs | **2,81** | 684 | 2,71 |
| 1,5 | 323 obs | **1,28** | 305 | 1,21 |
| 2,0 | 187 obs | **0,74** | 173 | 0,69 |

Esto es solo para **una** estrategia. Con selección, la exigencia sube (§5.5).

### 5.4 Sharpe deflactado (DSR)

Aquí está el corazón del asunto. El DSR es el PSR evaluado contra un umbral `SR*` que **no es cero**,
sino el Sharpe máximo que uno esperaría obtener por puro azar tras `N` pruebas:

```
DSR = PSR(SR_star_0)
```

con

```
SR_star_0 = sqrt(V) * [ (1 - gamma) * Zinv(1 - 1/N)  +  gamma * Zinv(1 - 1/(N*e)) ]
```

donde:

- `gamma` (`γ`) = constante de Euler–Mascheroni ≈ **0,5772156649**
- `e` = número de Euler ≈ 2,718281828
- `Zinv` (`Z⁻¹`) = función cuantil (inversa de la CDF) normal estándar
- `V = V[{SR_n}]` = **varianza de los Sharpe (por observación) de las `N` pruebas realizadas**;
  nótese que en la fórmula entra su raíz, `sqrt(V)`, es decir la desviación típica entre pruebas
- `N` = número de pruebas **efectivamente independientes**
- `SR_star_0` queda en unidades **por observación**, igual que `SR_hat`

> **⚠ Corrección a la transcripción más difundida.** Buena parte de las reproducciones online escriben
> el segundo término como `Z⁻¹[1 − 1/(N·e⁻¹)]`, es decir `N/e`. **Es incorrecto.** Verificado contra
> Monte Carlo del máximo de `N` normales estándar:

| `N` | fórmula con `N·e` | fórmula con `N/e` | Monte Carlo |
|---|---|---|---|
| 10 | 1,5746 | 0,8924 | **1,5409** |
| 50 | 2,2763 | 1,7941 | **2,2468** |
| 100 | 2,5306 | 2,0941 | **2,5058** |
| 1.000 | 3,2551 | 2,9111 | **3,2422** |
| 10.000 | 3,8607 | 3,5685 | **3,8535** |
| 100.000 | 4,3908 | 4,1328 | **4,3850** |

La forma con `N·e` reproduce el Monte Carlo con un error < 2,5 % en todo el rango y recupera el valor
canónico del paper (3,26 para `N = 1.000`). La forma con `N/e` subestima sistemáticamente entre un
11 % y un 43 %, siempre en la dirección de **aceptar estrategias falsas**.

**Cuánto deflacta.** Estrategia con 5 años de datos diarios (`T = 1.260`), Sharpe anualizado 1,0
(`t` clásico = 2,24), asimetría −0,5, curtosis 8:

| `N` pruebas | `sd(SR)` entre pruebas = 0,5 anual | `sd(SR)` entre pruebas = 1,0 anual |
|---|---|---|
| 1 | `SR*`=0,00 → **DSR = 0,9859** | `SR*`=0,00 → **DSR = 0,9859** |
| 10 | `SR*`=0,79 → DSR = 0,6796 | `SR*`=1,57 → DSR = 0,1038 |
| 100 | `SR*`=1,27 → DSR = 0,2803 | `SR*`=2,53 → DSR = 0,0004 |
| 1.000 | `SR*`=1,63 → DSR = 0,0843 | `SR*`=3,26 → DSR ≈ 0 |
| 10.000 | `SR*`=1,93 → DSR = 0,0206 | `SR*`=3,86 → DSR ≈ 0 |

Una estrategia con `PSR(0) = 0,986` (aparentemente sólida al 98,6 %) cae a **DSR = 0,28** con solo
100 pruebas y dispersión moderada. **Con 100 pruebas, el Sharpe 1,0 a 5 años no es evidencia de
nada.**

### 5.5 Cómo contar `N` honestamente

`N` es el parámetro que todo el mundo falsea, casi siempre sin querer. `N` **no** es "el número de
backtests que guardé". Incluye:

1. Cada combinación de hiperparámetros probada (ventanas, quantiles, umbrales, esquemas de
   ponderación, frecuencias de rebalanceo).
2. Cada variante de preprocesado (winsorización al 1 % vs 2,5 %, neutralizar por sector sí/no).
3. Cada decisión tomada **después de ver un resultado**, incluida invertir el signo de una señal.
4. Las pruebas realizadas por **otros agentes** del mismo repositorio sobre el mismo panel.
5. Las pruebas implícitas en la literatura que inspiró el factor.

> **Contrato del repo:** cada backtest ejecutado escribe una línea en un registro append-only
> (`data/trials/<familia>.jsonl`) con `{timestamp, family, config_hash, config, sharpe, ic}`. El `N`
> del DSR se lee **de ese registro**, nunca se elige a mano. `V̂` se calcula como la varianza muestral
> de los Sharpe registrados de esa familia. Un DSR calculado con un `N` inventado es peor que no
> calcular DSR, porque da una falsa sensación de rigor.

**Pruebas efectivamente independientes.** Si se prueban 500 configuraciones que son variaciones
mínimas unas de otras, `N = 500` sobre-penaliza. López de Prado y Lewis (2019) proponen agrupar
(clustering) las series de retornos de las pruebas y usar el **número de clusters** como `N` efectivo.
Regla práctica del repo, más simple y defendible:

```
N_eff = número de clusters con correlación media intra-cluster > 0,9
        sobre la matriz de correlaciones de los retornos de las pruebas
```

y en caso de duda se reporta el DSR con `N_eff` **y** con `N` bruto. Si las conclusiones difieren, la
señal está en zona gris y no se promociona.

### 5.6 Longitud mínima de backtest (MinBTL)

Bailey, Borwein, López de Prado y Zhu (2014). Complementaria del DSR: en vez de preguntar "¿es
significativo?", pregunta "¿cuántos años de datos necesito para que `N` pruebas no produzcan un
Sharpe 1,0 espurio?".

Como `SE(SR_anual) ≈ 1/√y` con `y` años de datos, el máximo esperado bajo `H₀` es
`E[max SR_anual] ≈ E[max_N]/√y`. Igualando a un `SR` objetivo:

```
MinBTL(años) = ( E[max_N] / SR_objetivo )²      con la cota clásica  E[max_N] ≤ √(2·ln N)
                                                 ⟹  MinBTL ≲ 2·ln(N) / SR_objetivo²
```

Calculado (`SR_objetivo = 1,0`):

| `N` | `E[max_N]` exacto | `√(2 ln N)` | MinBTL exacto (años) | `2 ln N` (años) |
|---|---|---|---|---|
| 10 | 1,5746 | 2,1460 | 2,48 | 4,61 |
| 100 | 2,5306 | 3,0349 | 6,40 | 9,21 |
| 1.000 | 3,2551 | 3,7169 | 10,60 | 13,82 |
| 10.000 | 3,8607 | 4,2919 | 14,90 | 18,42 |

**Aplicado a este repo:** tenemos 29,6 años de composición del índice. Eso soporta, con el criterio
exacto, del orden de `N ≈ 10⁵` pruebas antes de que un Sharpe 1,0 sea indistinguible del ruido. Es una
holgura cómoda **siempre que se use toda la historia**. Un backtest de 5 años solo soporta `N ≈ 25`
pruebas. **Corolario: en este repositorio los backtests cortos están prohibidos como evidencia
principal.**

---

## 6. Sobreajuste de backtest y probabilidad de sobreajuste (PBO)

### 6.1 El fenómeno

El sobreajuste de backtest no es un caso particular del sobreajuste de regresión. La diferencia clave:
en un backtest, el investigador **selecciona la configuración con mejor rendimiento in-sample** entre
`N` candidatas. Aunque ninguna tenga habilidad, la ganadora tendrá rendimiento in-sample positivo por
construcción, y su rendimiento out-of-sample será, en media, **cero o negativo**.

El *hold-out* clásico no protege: en cuanto se mira el hold-out y se vuelve a iterar, se convierte en
in-sample. En un repositorio con varios agentes trabajando sobre el mismo panel durante semanas, el
hold-out se quema en días.

### 6.2 CSCV: validación cruzada combinatoriamente simétrica

Bailey, Borwein, López de Prado y Zhu (2017), *Journal of Computational Finance*. Algoritmo exacto:

1. Construir la matriz `M` de dimensión `T × N`: `M[t,n]` = retorno del periodo `t` de la
   configuración `n`. **Todas las columnas deben cubrir el mismo periodo.**
2. Partir las `T` filas en `S` subconjuntos **contiguos y disjuntos** de igual tamaño (`S` par).
   López de Prado sugiere `S = 16`.
3. Para cada una de las `C(S, S/2)` combinaciones `c` de `S/2` subconjuntos:
   a. `J_c` = concatenación de los subconjuntos elegidos → **entrenamiento (IS)**.
   b. `J̄_c` = complemento → **test (OOS)**.
   c. `n*_c = argmax_n SR_IS(n)` — la configuración ganadora in-sample.
   d. `ω̄_c = rango_OOS(n*_c) / (N + 1)` ∈ (0,1) — su rango relativo out-of-sample.
   e. `λ_c = ln[ ω̄_c / (1 − ω̄_c) ]` — el logit del rango.
4. **PBO = P[λ ≤ 0] = (1/|C|) · Σ_c 1{λ_c ≤ 0}**

Interpretación: PBO es la probabilidad de que la configuración elegida in-sample quede **por debajo de
la mediana** out-of-sample. `PBO ≈ 0,5` significa que la selección in-sample no tiene ningún valor
predictivo: puro sobreajuste.

La simetría (evaluar todas las combinaciones, no una partición) es lo que hace el estimador
insesgado; usar solo la partición "primera mitad / segunda mitad" da una única observación de `λ`.

### 6.3 Calibración medida

Verificación con paneles de estrategias sin habilidad (`T = 1.008`, `N = 40`, σ = 1 %/día):

| `S` | combinaciones `C(S,S/2)` | PBO medio | sd | p5 | p95 |
|---|---|---|---|---|---|
| 10 | 252 | **0,496** | 0,152 | 0,272 | 0,743 |
| 12 | 924 | **0,494** | 0,164 | 0,258 | 0,722 |

El estimador está **bien calibrado**: converge a 0,50 bajo ruido puro, como debe.

**Pero su desviación típica es ≈ 0,16.** Un PBO de una sola ejecución no es un punto, es una
estimación con una banda de ±0,3 al 90 %. Reportarlo con tres decimales y sin banda es engañoso.

Barrido de habilidad (1 de 40 estrategias con alfa real creciente):

| alfa (pb/día) | Sharpe anual verdadero | PBO |
|---|---|---|
| 0 | 0,00 | 0,516 |
| 3 | 0,48 | 0,476 |
| 5 | 0,79 | 0,413 |
| 8 | 1,27 | 0,335 |
| 12 | 1,90 | 0,129 |
| 20 | 3,17 | 0,005 |

**Lectura clave para calibrar el umbral del repo:** el PBO solo baja de 0,35 cuando la estrategia
tiene un Sharpe verdadero superior a ≈ 1,3. Y solo baja de 0,15 con Sharpe > 1,9. Un umbral de
`PBO ≤ 0,10` sería **inalcanzable** para cualquier alfa realista en acciones (Sharpe 0,5–1,0) y
descartaría señales buenas. Un umbral de 0,50 no filtra nada.

> El umbral operativo debe ser **`PBO ≤ 0,35`** para promocionar, con la banda de confianza reportada.

### 6.4 Métricas complementarias del CSCV

Además del PBO, el mismo barrido produce:

- **Degradación del rendimiento**: regresión de `SR_OOS` sobre `SR_IS` a través de las combinaciones.
  La pendiente debería ser positiva; una pendiente **negativa** es la firma canónica del sobreajuste.
- **Probabilidad de pérdida**: `P[SR_OOS < 0]` de la configuración elegida IS.
- **Ratio de degradación**: `mediana(SR_OOS) / mediana(SR_IS)`.

---

## 7. Comparaciones múltiples

### 7.1 Las tres familias de control

Sean `M` contrastes con p-valores `p_1, …, p_M`, ordenados `p_(1) ≤ … ≤ p_(M)`.

**Bonferroni (FWER, un paso).** Rechaza `H_(i)` si `p_(i) ≤ α/M`. Equivalente:
`p_adj^(i) = min(M·p_(i), 1)`.
Controla la probabilidad de **cualquier** falso positivo. Muy conservador si los contrastes están
correlacionados (que es exactamente nuestro caso: los factores fundamentales están fuertemente
correlacionados entre sí).

**Holm (FWER, secuencial descendente).** Rechaza mientras `p_(i) ≤ α/(M − i + 1)`, parando en el
primer fallo. Domina uniformemente a Bonferroni: nunca rechaza menos, a veces más, con el mismo
control de FWER. **No hay razón para usar Bonferroni en vez de Holm** salvo para reportar el umbral
más conservador como referencia.

**Benjamini–Hochberg (FDR, escalonado ascendente).** Encuentra
`k = max{ i : p_(i) ≤ (i/M)·α }` y rechaza `H_(1) … H_(k)`. P-valor ajustado:

```
p_adj^(i) = min_{j ≥ i} min( (M/j)·p_(j) , 1 )
```

Controla la **proporción esperada de falsos positivos entre los rechazos**, no la probabilidad de
tener alguno. Válido bajo independencia y bajo dependencia positiva por regresión (PRDS), condición
que se cumple razonablemente en una familia de factores correlacionados positivamente.

**Benjamini–Yekutieli (FDR bajo dependencia arbitraria).** Sustituye `α` por `α/c(M)`:

```
c(M) = Σ_{j=1}^{M} 1/j ≈ ln(M) + γ + 1/(2M)          (γ = Euler–Mascheroni)
```

Umbral: `p_(i) ≤ (i / (M·c(M)))·α`. Es el que usan Harvey–Liu–Zhu.

**Romano–Wolf (FWER, escalonado descendente con bootstrap).** El más potente de los que controlan
FWER, porque **estima la dependencia entre contrastes por bootstrap** en vez de asumir el peor caso.
Procedimiento: se remuestrean conjuntamente todas las series de retorno (bootstrap por bloques,
§8), se calcula el **máximo** del estadístico studentizado en cada réplica, y ese máximo define la
distribución nula conjunta. Después se elimina el más significativo y se repite sobre el resto
(*stepdown*).

> **Para una familia de factores correlacionados como la de este repo, Romano–Wolf es
> estrictamente mejor que Bonferroni/Holm**: gana potencia sin perder control de FWER. Es también el
> más caro computacionalmente (`B` × `M` evaluaciones). Se exige en la promoción final, no en el
> cribado.

### 7.2 Umbrales calculados

Con `α = 0,05` bilateral, aproximación normal:

| `M` | `t` Bonferroni | `c(M)` | `t` BHY (rango 1) | `t` BH (rango `M`) |
|---|---|---|---|---|
| 10 | 2,807 | 2,929 | 3,137 | 1,960 |
| 20 | 3,023 | 3,598 | 3,392 | 1,960 |
| 50 | 3,291 | 4,499 | 3,692 | 1,960 |
| 100 | 3,481 | 5,187 | 3,900 | 1,960 |
| **316** | **3,778** | 6,335 | 4,215 | 1,960 |
| 1.000 | 4,056 | 7,485 | 4,504 | 1,960 |

**Cross-check independiente:** Harvey–Liu–Zhu reportan un umbral Bonferroni de **3,78** para su
conjunto de 316 factores. Mi cálculo desde cero da **3,778**. La aritmética reproduce su tabla, lo que
valida el resto de la columna.

Nótese la columna BH: en el rango peor (i=1) BH coincide con Bonferroni, pero en el rango `M`
el umbral es simplemente `α` (t = 1,96). BH es adaptativo: cuanto más rechazos hay, más permisivo se
vuelve. Por eso es la elección correcta para **cribar** una familia grande de factores, y Romano–Wolf
o Holm para **confirmar** los supervivientes.

### 7.3 El ajuste de Harvey–Liu–Zhu: `t > 3`

Harvey, Liu y Zhu (2016), *Review of Financial Studies* 29(1), 5-68. Recopilan 316 factores publicados
y argumentan que, dada la magnitud del *data mining* acumulado en la profesión, **un factor nuevo debe
superar `t > 3,0`** para ser creíble. Con el criterio `|t| > 3`, solo **9 de 313** variables
correlacionadas con el retorno sobreviven.

Su marco corrige además por las pruebas **no observadas**: los factores probados y nunca publicados.
Estiman ese número y lo incorporan al `M` efectivo.

**Matización importante, y es una crítica cuantitativa, no retórica.** El p-valor bilateral de
`t = 3,0` es `0,0027`. Bajo Bonferroni con `α = 0,05`, eso equivale a **`M = 18,5` pruebas**:

```
M_equivalente = α / p(t=3) = 0,05 / 0,0027 = 18,5
```

Es decir, **`t > 3` es el umbral Bonferroni de apenas 19 pruebas**. Con las 316 pruebas de su propio
conjunto, Bonferroni exigiría `t > 3,78`. El `3,0` de HLZ **no es un umbral conservador**: es su
resultado bajo BHY (control de FDR, no de FWER), que es sustancialmente más permisivo. Presentarlo
como "el listón alto" es engañoso: es el listón **mínimo**.

**Contrapunto obligatorio.** Chen y Zimmermann (*Review of Asset Pricing Studies*, 2020) estiman el
sesgo de publicación directamente sobre 156 carteras de predictores publicados y encuentran que los
retornos ajustados por sesgo son **solo un 12 % menores** que los in-sample, y que un umbral tan bajo
como `t > 1,8` controla el *multiple testing* entre los predictores que sobreviven a la revisión por
pares. Su argumento: la dispersión de retornos entre predictores es demasiado grande para explicarse
por ruido *data-mined*.

> **Postura del repo.** No se resuelve la disputa por decreto; se toma la posición conservadora y se
> declara la incertidumbre:
> - Factor **nuevo** (sin literatura previa): `t_NW > 3,0` **y** superar BH al 10 % dentro de su
>   familia. El `3,0` como suelo, con Romano–Wolf sobre la familia para la promoción final.
> - Factor **con literatura previa robusta** (SUE, PEAD, accruals, Piotroski): `t_NW > 2,0` basta,
>   porque la hipótesis no fue generada por estos datos. Este es precisamente el argumento bayesiano
>   de HLZ: el prior importa.
> - En ambos casos se reporta el p-valor ajustado por BH **y** por Romano–Wolf.

### 7.4 Aplicación al panel con solapamiento

Los p-valores que entran en cualquier procedimiento de multiplicidad deben venir **ya corregidos por
dependencia** (§3, §4). Meter en BH un conjunto de p-valores calculados con `t` ingenuos sobre
retornos solapados es aplicar una corrección exquisita a números que están inflados ×4. El orden
correcto es siempre:

```
1. corregir dependencia (Newey-West / Kolari-Pynnonen / bootstrap por bloques)
2. obtener p-valores validos
3. corregir multiplicidad (BH para cribar; Romano-Wolf o Holm para confirmar)
4. deflactar el Sharpe (DSR) con el N del registro de pruebas
```

Los pasos 3 y 4 **no son redundantes**: el 3 controla la tasa de error sobre una familia declarada
de contrastes; el 4 corrige la selección del máximo, incluyendo las pruebas que nunca llegaron a
generar un p-valor formal. Este repo exige los dos.

---

## 8. Validación cruzada purgada con embargo, y CPCV

### 8.1 Por qué la CV estándar no vale

La `KFold` de scikit-learn asume observaciones intercambiables. En un panel financiero con etiquetas
que abarcan `h` periodos, una observación de entrenamiento cuyo periodo de etiqueta solapa con el de
una observación de test comparte el **mismo retorno realizado**: el modelo ve la respuesta.

### 8.2 Purga

Sea `[t_{i,0}, t_{i,1}]` el **span de información** de la observación `i`. Se elimina del conjunto de
entrenamiento toda observación `i` tal que existe `j` en test con:

```
[t_{i,0}, t_{i,1}] ∩ [t_{j,0}, t_{j,1}] ≠ ∅
```

**El span no es solo la etiqueta.** Si las *features* miran hacia atrás `b` periodos y la etiqueta
mira hacia adelante `f`, el span de un evento en `T` es `[T − b, T + f]`. Para el ángulo B de este
repo, con `PreEventFeatures` en `[T−30, T−1]`, un modelo de mercado estimado en `[T−250, T−40]` y una
etiqueta `CAR[0, +60]`:

```
span = [T − 250, T + 60]   →   h_purga = 310 sesiones
```

Es un orden de magnitud más que el `post = 60` que uno pondría de forma ingenua. **Este es el número
que gobierna el coste de la CV y donde más se falla.**

### 8.3 Embargo

La purga no basta: una observación de entrenamiento **inmediatamente posterior** al test, aunque no
solape, está correlacionada con él por la persistencia serial del mercado. El embargo elimina
adicionalmente las observaciones de entrenamiento cuyo span empieza en:

```
( t_test_fin , t_test_fin + e ]
```

López de Prado recomienda `e ≈ 1 % de T` por fold. Con `T = 3.482` sesiones, `e ≈ 35` sesiones.

**El embargo es asimétrico**: solo hacia adelante. Hacia atrás ya lo cubre la purga.

### 8.4 Coste medido de la CV purgada

Observaciones eliminadas ≈ `2·K·(h + e)` (dos fronteras por fold):

| `K` | `h` (span) | embargo | obs. purgadas | % del panel (T=3.482) |
|---|---|---|---|---|
| 5 | 5 | 5 | 100 | 2,9 % |
| 5 | 21 | 12 | 330 | 9,5 % |
| 5 | 63 | 35 | 980 | **28,1 %** |
| 5 | 63 | 63 | 1.260 | 36,2 % |
| 10 | 21 | 12 | 660 | 19,0 % |
| 10 | 63 | 35 | 1.960 | **56,3 %** |
| 10 | 63 | 63 | 2.520 | 72,4 % |

**Lecturas operativas:**

1. **Más folds no es gratis.** Pasar de `K=5` a `K=10` con `h=63` duplica la purga y consume el 56 %
   del panel. Con el span de 310 sesiones del ángulo B, `K=10` sería inviable.
2. Para el ángulo B, **la unidad de partición no debe ser el día sino el trimestre fiscal**. Partiendo
   por trimestres (119 disponibles), un span de 310 sesiones ≈ 5 trimestres, y purgar 5 trimestres por
   frontera con `K=5` cuesta `2·5·5 = 50` de 119 trimestres (42 %). Sigue siendo caro pero manejable,
   y es la partición correcta conceptualmente.
3. Con `K=5`, `h=21` y embargo 12 (el caso típico del ángulo A) el coste es del 9,5 %: perfectamente
   asumible. **El ángulo A puede permitirse CV purgada estándar; el ángulo B necesita diseño.**

### 8.5 CPCV: validación cruzada purgada combinatoria

La CV purgada con `K` folds produce **una sola** trayectoria de backtest. La CPCV produce muchas:

1. Partir el panel en `N` grupos contiguos.
2. Usar `k` grupos como test (y `N−k` como entrenamiento), sobre **todas** las `C(N,k)` combinaciones.
3. Purgar y aplicar embargo en cada frontera.
4. Recombinar las predicciones OOS en trayectorias completas.

Número de trayectorias reconstruibles:

```
φ[N, k] = C(N, k) · k / N
```

| `N` | `k` | splits `C(N,k)` | trayectorias `φ[N,k]` | % entrenamiento |
|---|---|---|---|---|
| 6 | 2 | 15 | 5 | 67 % |
| 8 | 2 | 28 | 7 | 75 % |
| 10 | 2 | 45 | **9** | 80 % |
| 10 | 3 | 120 | 36 | 70 % |
| 12 | 2 | 66 | 11 | 83 % |
| 12 | 3 | 220 | 55 | 75 % |
| 16 | 2 | 120 | 15 | 88 % |
| 16 | 3 | 560 | 105 | 81 % |

**Confusión frecuente:** el número de *splits* (`C(N,k)`) no es el número de *trayectorias*
(`φ[N,k]`). Con `N=6, k=2` hay 15 splits pero solo 5 trayectorias.

**Por qué importa:** en vez de un único Sharpe OOS se obtiene una **distribución** de Sharpes, uno por
trayectoria. De ahí salen directamente el intervalo de confianza que exige `ARCHITECTURE.md` §3.8 y
una estimación honesta de la variabilidad del backtest.

> **Configuración recomendada para este repo:**
> - Ángulo A (factores, `T ≈ 7.500` sesiones): `N = 12`, `k = 2` → 66 splits, 11 trayectorias, 83 %
>   de entrenamiento. Coste de purga asumible con `h = 21`.
> - Ángulo B (eventos, 119 trimestres): partición **por trimestre fiscal**, `N = 10`, `k = 2` → 45
>   splits, 9 trayectorias. Purga de 5 trimestres por frontera.
> - En ambos casos: reportar mediana, p5 y p95 del Sharpe entre trayectorias.

### 8.6 Relación entre CPCV y PBO

Son complementarios y se confunden a menudo:

- **CPCV** responde: *dada esta configuración fija, ¿cuál es la distribución de su rendimiento OOS?*
- **PBO/CSCV** responde: *¿el proceso de seleccionar la mejor configuración entre `N` tiene algún
  valor predictivo?*

La CPCV valida **una** estrategia; el PBO valida **el procedimiento de búsqueda**. Este repo exige
ambos, porque una estrategia puede pasar CPCV brillantemente y aun así ser el producto de un proceso
de selección con PBO ≈ 0,5 (es decir, se encontró por suerte, y la siguiente vez que se busque saldrá
otra cosa).

---

## 9. Bootstrap estacionario de Politis–Romano

### 9.1 Algoritmo exacto

Politis y Romano (1994), *JASA* 89(428), 1303-1313. Remuestrea bloques de **longitud aleatoria
geométrica**, lo que preserva la estacionariedad de la serie remuestreada (el bootstrap por bloques de
longitud fija no lo hace).

Con `p ∈ (0,1]` y `L = 1/p` la longitud media de bloque, para cada réplica:

```
I_1 ~ Uniforme{1, …, T}
para t = 2, …, T:
    con probabilidad p:      I_t ~ Uniforme{1, …, T}      (nuevo bloque)
    con probabilidad 1 − p:  I_t = (I_{t−1} mod T) + 1     (continuar, envolvente)
serie remuestreada: x*_t = x_{I_t}
```

El **envolvente** (`mod T`) es esencial: sin él las observaciones del final de la muestra se
submuestrean y el estimador queda sesgado.

`p = 1` recupera el bootstrap iid.

### 9.2 Longitud de bloque

Politis y White (2004), con la corrección de Patton, Politis y White (2009), dan la longitud óptima
para la estimación de la varianza de la media:

```
b_opt = ( 2·Ĝ² / D̂_SB )^(1/3) · T^(1/3)
```

donde `Ĝ = Σ_k |k|·γ̂_k` y `D̂_SB = 2·(Σ_k γ̂_k)²` (el cuadrado de la varianza de largo plazo).
La dependencia `T^(1/3)` es la firma característica: **la longitud de bloque crece con la muestra**,
no es constante.

Calculada para un AR(1) con `T = 500`:

| `ρ` | `b_opt` |
|---|---|
| 0,2 | 4,4 |
| 0,4 | 7,7 |
| 0,6 | 12,1 |
| 0,8 | 21,5 |

### 9.3 Cobertura medida

Cobertura del intervalo de percentiles al 95 % para la media de un AR(1), `T = 500`, 300
simulaciones, `B = 400` réplicas. Nominal = 0,950.

| `ρ` | `L=1` | `L=3` | `L=5` | `L=10` | `L=20` | `L=40` | iid |
|---|---|---|---|---|---|---|---|
| 0,0 | 0,927 | 0,917 | 0,917 | 0,907 | 0,910 | 0,867 | 0,920 |
| 0,2 | 0,880 | 0,920 | **0,927** | 0,917 | 0,910 | 0,900 | 0,883 |
| 0,4 | 0,803 | 0,887 | **0,910** | **0,910** | 0,907 | 0,893 | 0,790 |
| 0,6 | 0,673 | 0,873 | 0,890 | **0,923** | 0,913 | 0,897 | 0,680 |
| 0,8 | 0,507 | 0,753 | 0,813 | 0,883 | 0,883 | **0,893** | 0,503 |

**Cuatro lecturas:**

1. **El bootstrap iid se desploma con dependencia.** Con `ρ = 0,8` su cobertura real es **0,503** para
   un nominal de 0,95: un intervalo "al 95 %" que acierta la mitad de las veces. Con `ρ = 0,6`, 0,680.
   `L = 1` reproduce el iid exactamente, como debe.
2. **La regla de Politis–White acierta el óptimo empírico.** Predice `b_opt` = 4,4 / 7,7 / 12,1 / 21,5
   para `ρ` = 0,2 / 0,4 / 0,6 / 0,8; el máximo empírico de cobertura cae en `L` = 5 / 5–10 / 10 / 40.
   La regla es utilizable tal cual.
3. **La sensibilidad a `L` es asimétrica**: pasarse de largo es barato (con `ρ=0,2`, `L=40` da 0,900
   frente al 0,927 óptimo), quedarse corto es caro (`L=1` da 0,880). **Ante la duda, `L` grande.**
4. **Honestidad sobre el nivel absoluto:** ninguna celda alcanza 0,950. Incluso con `ρ = 0` la
   cobertura es ≈ 0,92. El bootstrap de percentiles sub-cubre en muestras de `T = 500`; el efecto es
   de finita muestra y se atenúa con `T`. No hay que leer estas cifras como "el bootstrap falla", sino
   como "el bootstrap de percentiles es ligeramente liberal, y el iid es catastróficamente liberal".

> **Regla del repo:** para intervalos de confianza sobre series de retornos o de IC se usa siempre
> bootstrap estacionario, nunca iid. `L` se estima con Politis–White sobre la propia serie, con un
> suelo de `L ≥ h` cuando hay solapamiento de horizonte `h`, y se redondea **hacia arriba**.
> `B ≥ 1.000` para intervalos, `B ≥ 5.000` para p-valores de cola. Para intervalos críticos se
> prefiere BCa o bootstrap-t sobre el de percentiles, dado el sesgo documentado en la lectura 4.

### 9.4 Aplicación al panel con solapamiento transversal

Aquí hay una decisión de diseño que se equivoca con frecuencia. Al remuestrear un panel hay que
decidir **qué es la unidad de remuestreo**:

- **Remuestrear fechas completas** (todos los activos de una fecha juntos): preserva la correlación
  transversal. **Es lo correcto para este repo.**
- **Remuestrear activos**: destruye la correlación transversal y produce intervalos demasiado
  estrechos, exactamente el error de §1.2(a).
- **Remuestrear observaciones (activo, fecha) individuales**: destruye ambas dependencias. Nunca.

Y sobre esa unidad "fecha", el bootstrap estacionario en bloques recoge además la dependencia
temporal. La combinación correcta es: **bloques estacionarios sobre el eje de fechas, con la sección
cruzada completa como átomo indivisible.**

Para el ángulo B, el átomo debe ser aún más grueso: **la fecha de evento con todos sus eventos**, o
directamente el trimestre fiscal completo, porque toda la temporada de resultados comparte
condiciones macro.

---

## 10. White Reality Check y SPA de Hansen

### 10.1 Qué problema resuelven

Los procedimientos de §7 corrigen la multiplicidad para una familia de contrastes **independientes o
con dependencia asumida**. El Reality Check y el SPA hacen algo distinto y complementario: contrastan
la hipótesis nula de que **la mejor de `L` estrategias no supera a un benchmark**, estimando por
bootstrap la distribución conjunta del máximo, con la dependencia real entre estrategias.

Es la pregunta directa que quiere responder este repo: *entre todos los factores que he probado,
¿el mejor bate de verdad al benchmark, o es el máximo esperado del ruido?*

### 10.2 White (2000): Reality Check

Sea `f_{l,t}` el diferencial de rendimiento del modelo `l` frente al benchmark en `t`
(p.ej. retorno del factor menos retorno del benchmark, o la diferencia de funciones de pérdida).

```
H₀ :  max_{l=1..L}  E[f_l]  ≤  0
```

Estadístico:

```
f̄_l = (1/T)·Σ_t f_{l,t}
V   = max_{l=1..L}  √T · f̄_l
```

Bootstrap (estacionario, §9), para `b = 1..B`:

```
V*_b = max_{l=1..L}  √T · ( f̄*_{l,b} − f̄_l )
```

El **recentrado** `− f̄_l` es lo que impone la nula. P-valor:

```
p_RC = (1/B) · Σ_b 1{ V*_b > V }
```

**Debilidad conocida:** el RC usa la configuración menos favorable (todos los `E[f_l] = 0`). Si la
familia contiene muchas estrategias claramente **malas**, éstas no cambian `V` pero sí engordan la
cola de `V*`, y el test pierde potencia dramáticamente. En este repo, donde se probarán decenas de
variantes y muchas serán obviamente inferiores, esa debilidad es determinante.

### 10.3 Hansen (2005): Superior Predictive Ability

Hansen (*JBES* 23, 365-380) corrige las dos debilidades del RC:

**(1) Studentización.** Divide por el error estándar de cada estrategia, reduciendo la influencia de
las estrategias erráticas de alta varianza:

```
T^SPA = max( 0 ,  max_l  √T · f̄_l / ω̂_l )
```

donde `ω̂_l²` es un estimador consistente de la varianza asintótica de `√T·f̄_l` (HAC o bootstrap).

**(2) Distribución nula dependiente de la muestra.** En vez de recentrar todo a cero, recentra solo
las estrategias que no son "claramente inferiores":

```
g_l = f̄_l · 1{ f̄_l ≥ −A_l }
```

con el umbral `A_l = (1/4)·T^(−1/4)·ω̂_l` en la versión *consistente*. Hansen define tres variantes:
`A_l = 0` (cota superior, más conservadora), la consistente, y
`A_l = ω̂_l·√(2·ln ln T / T)` (cota inferior, más liberal). **Reportar las tres** es la práctica
recomendada: si las tres coinciden, la conclusión es robusta.

Bootstrap:

```
Z*_{l,b} = √T · ( f̄*_{l,b} − g_l ) / ω̂_l
T*_b     = max( 0 , max_l Z*_{l,b} )
p_SPA    = (1/B) · Σ_b 1{ T*_b > T^SPA }
```

> **Regla del repo:** se usa **SPA**, no RC, y se reportan las tres variantes de `A_l`. El RC se
> calcula solo como referencia histórica. El remuestreo es bloque estacionario sobre fechas completas
> (§9.4), nunca iid.

### 10.4 Cómo definir el benchmark en este repo

El SPA es tan informativo como su benchmark. Benchmarks apropiados, en orden creciente de exigencia:

1. **Cero** (¿el factor gana dinero?). Débil: cualquier exposición a beta lo pasa.
2. **Buy-and-hold del índice** con la misma exposición neta.
3. **Los factores canónicos** (mercado, tamaño, valor, momentum, calidad, inversión: Fama–French
   5F + momentum). Esto contrasta si el factor aporta **alfa incremental**.
4. **La mejor combinación ya viva en el repo.** El más exigente y el más honesto: ¿este factor nuevo
   añade algo a lo que ya tenemos?

Para promocionar una señal a "viva", este repo exige el nivel **3** como mínimo, y el nivel 4 para
incorporarla al modelo combinado.

---

## 11. Métricas de cartera: spreads por quintil

### 11.1 Construcción

Para cada fecha de rebalanceo `t`, se ordena `U_t` por la señal y se forman `Q` carteras (por defecto
`Q = 5`). El spread es:

```
spread_t = r_{Q5, t} − r_{Q1, t}
```

Decisiones que **deben declararse siempre**, porque cambian el resultado:

- **Ponderación**: equiponderada (dominada por small caps y con mayor coste) o por capitalización
  (dominada por megacaps y con menos dispersión). Reportar **ambas**. Una señal que solo funciona
  equiponderada es una señal de tamaño.
- **Neutralización**: quintiles globales o dentro de sector. Si la señal solo funciona globalmente, es
  una apuesta sectorial.
- **Rebalanceo**: `W-FRI` por defecto (`ARCHITECTURE.md` §3.7). Cambiar la frecuencia es una prueba
  más para el `N` del DSR.
- **Retardo de ejecución**: la señal de `t` se ejecuta con el precio de `t+1` como mínimo. Ejecutar al
  cierre de `t` con datos de `t` es look-ahead.

### 11.2 Contrastes

**(a) Significancia del spread.** `t_NW` sobre la serie `{spread_t}` con la regla de lags de §3.2.
El intervalo de confianza por bootstrap estacionario (§9) es el que manda si difiere del de NW.

**(b) Monotonicidad.** Un `Q5 − Q1` significativo es compatible con un patrón no monótono
(p.ej. solo el `Q1` funciona). Patton y Timmermann (2010, *JFE* 98(3), 605-625) proponen el
**test MR**:

```
H₀ :  Δ_j = μ_{j+1} − μ_j  ≤ 0   para todo j        (no hay relación creciente)
H₁ :  Δ_j > 0 para todo j                            (monotonía estricta)

J = min_{j=1..Q−1}  Δ̂_j
```

y se obtiene el p-valor por bootstrap por bloques bajo la nula. El estadístico es el **mínimo** de las
diferencias: exige que *todos* los escalones vayan en el sentido correcto. Es deliberadamente
exigente, y por eso es informativo.

Su propio paper documenta que muchos patrones aceptados en la literatura **no** superan el MR: el
`Q5−Q1` es significativo pero la relación no es monótona.

**(c) Contribución de cada pata.** Reportar por separado `Q5 − mediana` y `mediana − Q1`. Ambas deben
tener el signo correcto. Una señal cuyo alfa está entero en la pata corta tiene un problema de
implementabilidad severo (coste de préstamo, disponibilidad, riesgo de *short squeeze*), que el
`CostModel` debe capturar.

**(d) Fama–MacBeth como control paramétrico.** Regresión transversal por fecha del retorno sobre la
señal más controles (tamaño, beta, sector), y luego `t_NW` sobre la serie temporal de los
coeficientes. Es la versión continua del *sort*, aprovecha toda la sección cruzada y controla por
covariables de forma limpia. Debe coincidir cualitativamente con el *sort*; si no, hay no linealidad
y el *sort* está capturando algo que la regresión lineal no.

### 11.3 Potencia medida

Simulación con `T = 1.260` fechas (5 años diarios), `n = 475` activos, quintiles, **residuos
transversalmente independientes** y sin solapamiento:

| IC verdadera | `t` medio del spread `Q5−Q1` | potencia (\|t\|>1,96) |
|---|---|---|
| 0,00 | −0,09 | 0,050 |
| 0,01 | 6,75 | 1,000 |
| 0,02 | 13,60 | 1,000 |
| 0,03 | 20,44 | 1,000 |
| 0,05 | 34,15 | 1,000 |

La calibración bajo la nula es perfecta (`t` medio ≈ 0, rechazo 0,050). Pero **estas cifras son
irrealmente optimistas y hay que decirlo**: la simulación no tiene factor común. En un mercado real,
los residuos de las 475 acciones comparten exposición a sector y estilo, y aunque el `Q5−Q1` cancela
el beta de mercado, no cancela esa correlación residual.

Corrigiendo con la breadth efectiva de §11.4 (`ρ = 0,05` ⟹ `BR_eff = 19,2` de 475, factor de
corrección `√(475/19,2) = 4,97`):

| IC verdadera | `t` bruto (5 años) | `t` corregido (5 años) | `t` corregido (15 años) | `t` corregido (30 años) |
|---|---|---|---|---|
| 0,01 | 6,75 | 1,36 | 2,35 | 3,32 |
| 0,02 | 13,60 | **2,73** | **4,73** | **6,69** |
| 0,03 | 20,44 | 4,11 | 7,12 | 10,07 |

**Estos números sí son creíbles** y coinciden con la experiencia publicada: un factor con rank IC de
0,02 no es concluyente con 5 años y sí lo es con 15.

**Comprobación de coherencia interna del repo.** §14.1 exige rank IC ≥ 0,020, `t_NW` ≥ 3,0 y ≥ 15 años
de muestra. La tabla dice que esa combinación produce `t` ≈ 4,7 — **holgadamente por encima del
umbral**, incluso tras aplicar el descuento del 25 % por el sesgo residual de Newey–West (§3.3), que
lo deja en ≈ 3,8. Los tres umbrales son mutuamente consistentes y alcanzables. No son un listón
arbitrario.

**Conclusión que reorienta el problema.** La potencia estadística bruta **no es el cuello de botella
de este proyecto** una vez se usa la historia completa. Los cuellos de botella son, en este orden:

1. La dependencia, temporal y transversal (inflación ×5–8 del `t`) — §3, §4.
2. La selección entre muchas pruebas (DSR, PBO) — §5, §6.
3. Los costes de transacción, que no aparecen en ningún `t`.

Un equipo que dedique su esfuerzo a "conseguir más datos" está optimizando la variable equivocada;
uno que use 5 años en vez de los 30 disponibles se está disparando en el pie.

### 11.4 Traducción IC → Sharpe: la ley fundamental y su trampa

Grinold (1989), extendida por Clarke, de Silva y Thorley (2002):

```
IR = IC · √BR · TC
```

con `BR` = número de apuestas **independientes** por año y `TC` = *transfer coefficient*, la
correlación transversal entre las posiciones ideales y las realmente tomadas tras restricciones
(límites de peso, `max_weight = 0,02`, `adv_participation = 0,05`, prohibición de cortos...).

Aplicado ingenuamente con `BR = 52 × 475 = 24.700`:

| IC | rebalanceo | `BR` | `IR` (TC=1) |
|---|---|---|---|
| 0,02 | semanal | 24.700 | **3,14** |
| 0,03 | semanal | 24.700 | **4,71** |

Números absurdos. Nadie obtiene IR de 4,7 con una IC de 0,03. **El error está en `BR`**: las 475
apuestas no son independientes, comparten factores de mercado y sector. Con la misma álgebra de §1.2:

```
BR_eff = N / [1 + (N−1)·ρ]
```

| `ρ` entre apuestas | `BR_eff` (de 475) | `IR` con IC=0,03, 52 reb/año |
|---|---|---|
| 0,00 | 475,0 | 4,71 |
| 0,05 | 19,2 | **0,95** |
| 0,10 | 9,8 | **0,68** |
| 0,20 | 5,0 | 0,48 |
| 0,30 | 3,3 | 0,39 |

Con `ρ = 0,05` (modesta) el IR cae de 4,71 a **0,95**, que sí es una cifra creíble para un factor
bueno. Y aplicando un `TC = 0,5` realista tras restricciones, queda en ≈ 0,48.

> **Regla de coherencia del repo:** si el Sharpe del backtest supera en más de un factor 2 el que
> predice `IC · √BR_eff · TC` con `ρ` estimado de los datos, **hay un error en el backtest** (look-ahead,
> costes ausentes, sesgo de supervivencia). Es un test de sanidad barato que atrapa la mayoría de los
> fallos graves antes que cualquier test estadístico.

---

## 12. Reproducción de las verificaciones

Todas las cifras de este documento proceden de simulaciones deterministas con semilla fija
(20260804, 7, 11, 3). Ninguna cifra sin marca ⚠ está copiada de un abstract: o es cálculo directo, o
es Monte Carlo reproducible.

| Sección | Qué verifica | Método | Resultado clave |
|---|---|---|---|
| §1.2 | inflación por correlación transversal | 20.000 réplicas, Cholesky, `n`=500 | teoría 5,094 / medido 5,081 |
| §3.2 | regla de lags NW | cálculo directo | `T`=7.500 ⟹ `L`=10 |
| §3.3 | inflación del `t` por solapamiento | 2.000 réplicas MA(`h−1`), `T`=2.500 | `h`=63 ⟹ ×7,98 ingenuo, ×1,25 NW |
| §5.1 | curtosis cruda vs exceso en PSR | identidad exacta con Lo (2002) | 1,00249688 en ambos lados |
| §5.1 | efecto de no-normalidad por frecuencia | cálculo directo | diaria +1,8 %, mensual +26,7 % |
| §5.3 | MinTRL | cálculo directo | `SR`=1,0 ⟹ 2,81 años |
| §5.4 | `E[max SR]` con `N·e` vs `N/e` | Monte Carlo, hasta 200.000 réplicas | `N`=1.000: 3,2551 vs MC 3,2422 |
| §5.6 | MinBTL | cálculo directo | `N`=1.000 ⟹ 10,6 años |
| §6.3 | calibración del PBO | 40 paneles independientes por `S` | ruido puro ⟹ 0,494 ± 0,164 |
| §7.2 | umbrales de multiplicidad | cálculo directo | `M`=316 ⟹ Bonferroni 3,778 (HLZ: 3,78) |
| §8.4 | coste de la purga | cálculo directo | `K`=10, `h`=63, `e`=35 ⟹ 56,3 % |
| §8.5 | `φ[N,k]` de CPCV | cálculo combinatorio | `φ[10,2]` = 9 |
| §9.2 | `b_opt` de Politis–White | cálculo directo | `ρ`=0,6, `T`=500 ⟹ 12,1 |
| §9.3 | cobertura del bootstrap | 300 sims × 400 réplicas por celda | `ρ`=0,8: iid 0,503 vs SB 0,893 |
| §11.3 | potencia del spread por quintil | 300 sims, `T`=1.260, `n`=475 | IC=0,02 ⟹ `t`=13,60 bruto |
| §11.4 | breadth efectiva | cálculo directo | `ρ`=0,05 ⟹ `BR_eff`=19,2 de 475 |
| §1.1 | aritmética del panel | `data/seed/sp500_historical_components.csv` | 3.482 snapshots, 1.128 tickers |

**Estos números son los casos de test de `earnings_alpha.stats`.** Una implementación correcta debe
reproducirlos:

| Función | Entrada | Salida esperada |
|---|---|---|
| `expected_max_sharpe` | `N=1000, mean=0, sd=1` | `3,2551` (±1e-4) |
| `expected_max_sharpe` | `N=100` | `2,5306` |
| `psr` | `SR=0,10, SR*=0, T=1260, γ₃=0, γ₄=3` | denominador `1,00249688` |
| `min_trl` | `SR_ann=1,0, γ₃=−0,5, γ₄=8, α=0,05` | `709` obs |
| `cpcv_paths` | `N=10, k=2` | `9` trayectorias, `45` splits |
| `cpcv_paths` | `N=6, k=2` | `5` trayectorias, `15` splits |
| `pbo_cscv` | panel de ruido `T=1008, N=40, S=12` | `≈0,50` (banda ±0,33 al 90 %) |
| `newey_west_lags` | `T=1260, h=1` | `7` |
| `newey_west_lags` | `T=1260, h=21` | `20` (domina el suelo `h−1`) |
| `bonferroni_t` | `M=316, α=0,05` | `3,778` |
| `effective_n` | `n=500, ρ̄=0,05` | `19,3` |

Un test de regresión que compruebe la tabla anterior detecta de golpe los errores 3, 4, 8 y 11 de
§15, que son los que fallan **en silencio y en la dirección peligrosa**.

---

## 13. Protocolo integrado de validación

El orden importa: cada etapa filtra antes de gastar cómputo en la siguiente.

### Etapa 0 — Higiene point-in-time (bloqueante)
Antes de cualquier estadístico. Si falla, nada de lo demás significa nada.

- [ ] Universo PIT en cada fecha (`universe.members_on`), sin supervivencia.
- [ ] Fundamentales indexados por `available_at`, no `period_end`.
- [ ] `tradable_date` correcta para BMO/AMC (`pit`); test explícito de que un evento AMC no es
      negociable el mismo día.
- [ ] Retardo de ejecución ≥ 1 sesión entre señal y precio.
- [ ] Ningún dato reexpresado (`is_restated = True`) en la serie de backtest.

### Etapa 1 — Cribado (barato)
- [ ] rank IC media y por año; fracción de meses positivos.
- [ ] `t_NW` con `L = max(⌊4(T/100)^(2/9)⌋, h−1)`.
- [ ] Perfil de decaimiento: IC frente al horizonte `h` ∈ {1, 5, 21, 63}.
- [ ] Test de coherencia de §11.4 (`IC · √BR_eff · TC` frente al Sharpe del backtest).

Se descartan aquí las señales con `t_NW < 2,0` o IC media con signo contrario al teórico.

### Etapa 2 — Estructura de la señal
- [ ] Spread `Q5−Q1` con `t_NW` e IC bootstrap.
- [ ] Monotonicidad (MR de Patton–Timmermann).
- [ ] Contribución de cada pata por separado.
- [ ] IC neutralizada por sector; equiponderada y por capitalización.
- [ ] Fama–MacBeth con controles.

### Etapa 3 — Robustez temporal
- [ ] CPCV (`N=12, k=2` ángulo A; `N=10, k=2` por trimestres ángulo B).
- [ ] Distribución de Sharpe entre trayectorias: mediana, p5, p95.
- [ ] Estabilidad por subperiodo (pre/post 2008, pre/post 2020).
- [ ] Estabilidad por régimen de volatilidad.

### Etapa 4 — Selección y multiplicidad
- [ ] PBO por CSCV sobre **todas** las configuraciones probadas de la familia.
- [ ] BH al 10 % sobre la familia; Romano–Wolf para los supervivientes.
- [ ] DSR con `N` y `V̂` leídos del registro de pruebas.
- [ ] SPA de Hansen contra el benchmark de nivel 3 (factores canónicos).

### Etapa 5 — Implementabilidad
- [ ] Rendimiento neto con `CostModel` completo (spread + impacto √participación + comisión + préstamo).
- [ ] Rotación y capacidad a `adv_participation = 0,05`.
- [ ] Concentración: peso máximo, número efectivo de posiciones.
- [ ] Sensibilidad al retardo de ejecución (1, 2, 3 sesiones): un alfa que muere con 2 días de retardo
      no es explotable.

---

## 14. Umbrales concretos exigidos por este repositorio

Tres estados: **VIVA** (se incorpora al modelo combinado), **CUARENTENA** (se sigue investigando, no
se opera), **MUERTA** (se archiva con su registro de pruebas).

### 14.1 Ángulo A — factores fundamentales cross-section

| # | Criterio | VIVA | CUARENTENA | Justificación |
|---|---|---|---|---|
| 1 | rank IC media (h=21) | ≥ 0,020 | ≥ 0,010 | §2.3: por debajo, ni con `n`=475 |
| 2 | `t_NW` de la IC | ≥ **3,0** (nuevo) / ≥ 2,0 (con literatura) | ≥ 2,0 / ≥ 1,7 | §7.3, HLZ vs Chen–Zimmermann |
| 3 | IC-IR anualizada | ≥ 0,30 | ≥ 0,20 | consistencia con IR realizable |
| 4 | Fracción de meses con IC > 0 | ≥ 55 % | ≥ 52 % | estabilidad, no solo media |
| 5 | `t_NW` del spread `Q5−Q1` **neto de costes** | ≥ **3,0** | ≥ 2,0 | §11.2 |
| 6 | Monotonicidad MR (p-valor) | ≤ 0,10 | ≤ 0,25 | §11.2(b) |
| 7 | Ambas patas con signo correcto | sí | `Q5−mediana` ≥ 0 | §11.2(c) |
| 8 | IC neutralizada por sector | ≥ 60 % de la IC cruda | ≥ 40 % | si no, es apuesta sectorial |
| 9 | **DSR** con `N` del registro | ≥ **0,95** | ≥ 0,90 | §5.4 |
| 10 | **PBO** (CSCV) | ≤ **0,35** | ≤ 0,50 | §6.3, calibrado |
| 11 | CPCV: p5 del Sharpe entre trayectorias | > 0 | > −0,25 | §8.5 |
| 12 | CPCV: mediana / Sharpe muestra completa | ≥ 0,50 | ≥ 0,30 | degradación aceptable |
| 13 | BH al 10 % dentro de la familia | pasa | pasa al 20 % | §7.1 |
| 14 | Romano–Wolf FWER 5 % | pasa | — | solo para VIVA |
| 15 | SPA de Hansen vs FF5+MOM | p ≤ **0,05** en las 3 variantes de `A_l` | p ≤ 0,10 en la consistente | §10.3 |
| 16 | IC bootstrap estacionario 95 % del Sharpe | excluye 0 | — | §9, mandato §3.8 |
| 17 | Longitud de muestra | ≥ 15 años | ≥ 10 años | §5.6, MinBTL |
| 18 | Supervivencia a 2 sesiones de retardo | ≥ 60 % del alfa | ≥ 40 % | implementabilidad |
| 19 | Coherencia `IC·√BR_eff·TC` | Sharpe backtest < 2× predicho | < 3× | §11.4, test de sanidad |

**Cualquier criterio marcado como bloqueante que falle → MUERTA**, sin excepción: 2, 9, 10, 15.
El resto en zona de CUARENTENA permite seguir investigando.

### 14.2 Ángulo B — eventos de resultados

| # | Criterio | VIVA | CUARENTENA | Justificación |
|---|---|---|---|---|
| 1 | Nº de eventos tras filtros | ≥ 10.000 | ≥ 5.000 | de los ~56.000 disponibles |
| 2 | **`n_eff`** tras corregir agrupamiento | ≥ **1.000** | ≥ 500 | §1.2(a): la cifra que manda |
| 3 | Eventos con fecha estimada (`is_estimated_date`) | excluidos | ≤ 10 % | §1: error de un día invalida |
| 4 | `t` del CAR: **Kolari–Pynnönen o cartera calendario** | ≥ **3,0** | ≥ 2,5 | §4.2; el `t` transversal simple no se acepta |
| 5 | Consistencia entre método (1) y (2) de §4.2 | `t` difieren < 30 % | < 50 % | si difieren mucho, `ρ̄` mal estimado |
| 6 | Purga con span completo `[T−250, T+60]` | sí | sí | §8.2, bloqueante |
| 7 | Partición de CV por trimestre fiscal | sí | sí | §8.4 |
| 8 | **DSR** | ≥ **0,95** | ≥ 0,90 | §5.4 |
| 9 | **PBO** | ≤ **0,35** | ≤ 0,50 | §6.3 |
| 10 | Estabilidad: signo consistente en ≥ 3 de 4 subperiodos de ~7 años | sí | 2 de 4 | los efectos de evento decaen |
| 11 | Sensibilidad al retardo BMO/AMC | alfa cae < 40 % al posponer 1 sesión | < 60 % | detecta look-ahead en `tradable_date` |
| 12 | Neto de costes con gap overnight modelado | `t` ≥ 2,5 | ≥ 2,0 | `EventBacktest` modela el gap |

### 14.3 Detector de negociación informada pre-evento

Este caso tiene un requisito adicional que **ninguna de las métricas anteriores cubre**: hay que
demostrar que el detector detecta. `data.synthetic.SyntheticMarket` genera un subconjunto configurable
de "eventos con filtración" con *ground truth* conocido (`ARCHITECTURE.md` §3.3). Eso permite medir
**potencia real**, no solo significancia:

| # | Criterio | VIVA | CUARENTENA |
|---|---|---|---|
| 1 | AUC sobre eventos sintéticos con filtración | ≥ 0,65 | ≥ 0,58 |
| 2 | Tasa de falsos positivos sobre eventos sintéticos **sin** filtración | ≤ 0,10 al umbral operativo | ≤ 0,15 |
| 3 | El detector **no** dispara por encima del azar cuando la filtración se desactiva | verificado | verificado |
| 4 | Precisión en el decil superior sobre datos reales | ≥ 2× la tasa base | ≥ 1,5× |
| 5 | Validación temporal: `SurpriseModel` con CV purgada + embargo | obligatorio | obligatorio |
| 6 | Todas las features derivan de datos públicos | verificado y documentado | bloqueante |

El criterio 3 es el que más se olvida: un detector que dispara siempre tiene AUC alta si la clase
positiva es frecuente. Hay que verificar el caso nulo explícitamente.

### 14.4 Cómo se declara muerta una señal

Simétrico y con la misma disciplina que la promoción, para evitar el sesgo de mantener vivo lo que
costó encontrar:

- El `t_NW` de la IC cae por debajo de 1,5 en una ventana móvil de 3 años **y** el DSR recalculado
  cae por debajo de 0,80.
- El SPA contra el modelo combinado vivo deja de rechazar (`p > 0,20`): la señal ya no aporta nada
  incremental.
- La rotación necesaria para mantener el alfa supera la capacidad a `adv_participation = 0,05`.
- Se descubre una violación point-in-time en su cadena de datos. **Muerte inmediata**, sin
  cuarentena, y se re-audita todo lo que comparta esa cadena.

---

## 15. Errores frecuentes: lista de comprobación

Ordenados por frecuencia con la que aparecen en este tipo de proyecto.

1. **`t` ingenuo sobre retornos solapados.** Infla ×√h. §3.3.
2. **Ignorar el agrupamiento de fechas de eventos.** Infla ×5 con `ρ̄`=0,05. §1.2(a).
3. **Curtosis en exceso en el PSR.** `scipy` devuelve exceso por defecto. §5.1.
4. **`E[max SR]` con `N/e` en vez de `N·e`.** Subestima el umbral un 11 %. §5.4.
5. **`N` del DSR inventado.** Sin registro de pruebas, el DSR es teatro. §5.5.
6. **Sharpe anualizado dentro de las fórmulas PSR/DSR.** Deben ser por observación. §5.
7. **Purgar solo la etiqueta, no el span completo de información.** §8.2.
8. **Confundir splits con trayectorias en CPCV.** `C(6,2)=15` splits pero `φ=5` trayectorias. §8.5.
9. **Bootstrap iid sobre series autocorrelacionadas.** Cobertura muy por debajo del nominal. §9.3.
10. **Remuestrear activos en vez de fechas.** Destruye la correlación transversal. §9.4.
11. **Corregir multiplicidad antes que dependencia.** El orden correcto está en §7.4.
12. **Reportar `Q5−Q1` sin comprobar monotonicidad.** §11.2(b).
13. **Usar la lista actual del S&P 500 para fechas pasadas.** Sesgo de supervivencia; el repo tiene
    la composición histórica precisamente para esto.
14. **Aplicar la ley fundamental con `BR` bruta.** Produce IR de 4,7 que nadie ha visto jamás. §11.4.
15. **Tratar el hold-out como sagrado tras haberlo mirado.** En cuanto se itera sobre él, es
    in-sample y entra en `N`.
16. **Confundir PBO con CPCV.** Validan cosas distintas; se exigen los dos. §8.6.

---

## 16. Referencias

**Coeficiente de información, spreads y ley fundamental**

- Grinold, R. C. (1989). *The Fundamental Law of Active Management*. **Journal of Portfolio
  Management**, 15(3), 30-37. https://joim.com/wp-content/uploads/emember/downloads/p0158.pdf
- Clarke, R., de Silva, H., y Thorley, S. (2002). *Portfolio Constraints and the Fundamental Law of
  Active Management*. **Financial Analysts Journal**, 58(5), 48-66.
  https://www.researchgate.net/publication/228182902_Portfolio_Constraints_and_the_Fundamental_Law_of_Active_Management
- Ding, Z., y Martin, R. D. (2017). *The Fundamental Law of Active Management: Redux*. **Journal of
  Empirical Finance**, 43, 91-114.
  https://www.sciencedirect.com/science/article/pii/S0927539817300543
- Patton, A. J., y Timmermann, A. (2010). *Monotonicity in Asset Returns: New Tests with Applications
  to the Term Structure, the CAPM, and Portfolio Sorts*. **Journal of Financial Economics**, 98(3),
  605-625. https://public.econ.duke.edu/~ap172/Patton_Timmermann_sorts_JFE_Dec2010.pdf
- Bonett, D. G., y Wright, T. A. (2000). *Sample size requirements for estimating Pearson, Kendall and
  Spearman correlations*. **Psychometrika**, 65, 23-28.

**Errores estándar robustos y datos solapados**

- Newey, W. K., y West, K. D. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*. **Econometrica**, 55(3), 703-708.
- Newey, W. K., y West, K. D. (1994). *Automatic Lag Selection in Covariance Matrix Estimation*.
  **Review of Economic Studies**, 61(4), 631-653.
- Andrews, D. W. K. (1991). *Heteroskedasticity and Autocorrelation Consistent Covariance Matrix
  Estimation*. **Econometrica**, 59(3), 817-858.
- Hansen, L. P., y Hodrick, R. J. (1980). *Forward Exchange Rates as Optimal Predictors of Future Spot
  Rates*. **Journal of Political Economy**, 88(5), 829-853.
- Britten-Jones, M., Neuberger, A., y Nolte, I. (2011). *Improved Inference and Estimation in
  Regression with Overlapping Observations*. **Journal of Business Finance & Accounting**.
  https://warwick.ac.uk/fac/soc/wbs/subjects/finance/faculty1/anthony_neuberger/improved.pdf
- Petersen, M. A. (2009). *Estimating Standard Errors in Finance Panel Data Sets: Comparing
  Approaches*. **Review of Financial Studies**, 22(1), 435-480.
- Driscoll, J. C., y Kraay, A. C. (1998). *Consistent Covariance Matrix Estimation with Spatially
  Dependent Panel Data*. **Review of Economics and Statistics**, 80(4), 549-560.
  http://fmwww.bc.edu/repec/bocode/x/xtscc_paper
- Fama, E. F., y MacBeth, J. D. (1973). *Risk, Return, and Equilibrium: Empirical Tests*. **Journal of
  Political Economy**, 81(3), 607-636.

**Estudios de evento con correlación cruzada**

- Kolari, J. W., y Pynnönen, S. (2010). *Event Study Testing with Cross-sectional Correlation of
  Abnormal Returns*. **Review of Financial Studies**, 23(11), 3996-4025.
  https://academic.oup.com/rfs/article-abstract/23/11/3996/1605665
- Kolari, J. W., Pape, B., y Pynnönen, S. (2018). *Event Study Testing with Cross-Sectional
  Correlation Due to Partially Overlapping Event Windows*.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167271
- Boehmer, E., Musumeci, J., y Poulsen, A. B. (1991). *Event-study methodology under conditions of
  event-induced variance*. **Journal of Financial Economics**, 30(2), 253-272.
- EventStudyTools. *Significance Tests: Patell Z & BMP*.
  https://www.eventstudytools.com/significance-tests

**Sharpe: PSR, DSR, MinTRL, MinBTL**

- Lo, A. W. (2002). *The Statistics of Sharpe Ratios*. **Financial Analysts Journal**, 58(4), 36-52.
  https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf
- Bailey, D. H., y López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier*. **Journal of
  Risk**, 15(2), 3-44. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
- Bailey, D. H., y López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection
  Bias, Backtest Overfitting and Non-Normality*. **Journal of Portfolio Management**, 40(5), 94-107.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 ·
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- Wikipedia. *Deflated Sharpe ratio*. https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio
- Opdyke, J. D. (2007). *Comparing Sharpe ratios: So where are the p-values?* **Journal of Asset
  Management**, 8, 308-336.
- López de Prado, M., y Lewis, M. J. (2019). *Detection of False Investment Strategies Using
  Unsupervised Learning Methods*. **Quantitative Finance**, 19(9), 1555-1565.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017

**Sobreajuste de backtest**

- Bailey, D. H., Borwein, J. M., López de Prado, M., y Zhu, Q. J. (2014). *Pseudo-Mathematics and
  Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance*.
  **Notices of the AMS**, 61(5), 458-471.
- Bailey, D. H., Borwein, J. M., López de Prado, M., y Zhu, Q. J. (2017). *The Probability of Backtest
  Overfitting*. **Journal of Computational Finance**, 20(4), 39-69.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 ·
  https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Purging, embargo,
  CPCV.) Resumen: https://en.wikipedia.org/wiki/Purged_cross-validation

**Comparaciones múltiples**

- Harvey, C. R., Liu, Y., y Zhu, H. (2016). *…and the Cross-Section of Expected Returns*. **Review of
  Financial Studies**, 29(1), 5-68. https://academic.oup.com/rfs/article/29/1/5/1843824 ·
  https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF
- Harvey, C. R., y Liu, Y. (2015). *Backtesting*. **Journal of Portfolio Management**, 42(1), 13-28.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489 ·
  https://www.cmegroup.com/content/dam/cmegroup/education/files/backtesting.pdf
- Chen, A. Y., y Zimmermann, T. (2020). *Publication Bias and the Cross-Section of Stock Returns*.
  **Review of Asset Pricing Studies**, 10(2), 249-289.
  https://academic.oup.com/raps/article-abstract/10/2/249/5640503
- Benjamini, Y., y Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and
  Powerful Approach to Multiple Testing*. **JRSS B**, 57(1), 289-300.
- Benjamini, Y., y Yekutieli, D. (2001). *The Control of the False Discovery Rate in Multiple Testing
  under Dependency*. **Annals of Statistics**, 29(4), 1165-1188.
- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. **Scandinavian Journal
  of Statistics**, 6(2), 65-70.
- Romano, J. P., y Wolf, M. (2005). *Stepwise Multiple Testing as Formalized Data Snooping*.
  **Econometrica**, 73(4), 1237-1282.
  http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/MultipleComparision/RomanoWolf05.pdf
- Romano, J. P., y Wolf, M. (2016). *Efficient Computation of Adjusted p-Values for Resampling-Based
  Stepdown Multiple Testing*. **Statistics & Probability Letters**, 113, 38-40.
  https://www.econstor.eu/bitstream/10419/162422/1/econwp219.pdf
- Clarke, D., Romano, J. P., y Wolf, M. (2020). *The Romano–Wolf Multiple-Hypothesis Correction in
  Stata*. **The Stata Journal**, 20(4), 812-843. https://docs.iza.org/dp12845.pdf

**Bootstrap y comparación de modelos**

- Politis, D. N., y Romano, J. P. (1994). *The Stationary Bootstrap*. **JASA**, 89(428), 1303-1313.
- Politis, D. N., y White, H. (2004). *Automatic Block-Length Selection for the Dependent Bootstrap*.
  **Econometric Reviews**, 23(1), 53-70.
- Patton, A., Politis, D. N., y White, H. (2009). *Correction to "Automatic Block-Length Selection for
  the Dependent Bootstrap"*. **Econometric Reviews**, 28(4), 372-375.
- White, H. (2000). *A Reality Check for Data Snooping*. **Econometrica**, 68(5), 1097-1126.
  https://users.ssc.wisc.edu/~bhansen/718/White2000.pdf
- Hansen, P. R. (2005). *A Test for Superior Predictive Ability*. **Journal of Business & Economic
  Statistics**, 23(4), 365-380. https://cdr.lib.unc.edu/downloads/zp38wf793
- Corradi, V., y Swanson, N. R. (2011). *The White Reality Check and Some of Its Recent Extensions*.
  https://econweb.rutgers.edu/nswanson/papers/corradi_swanson_whitefest_1108_2011_09_06.pdf
- Hsu, P.-H., Hsu, Y.-C., y Kuan, C.-M. (2010). *Testing the Predictive Ability of Technical Analysis
  Using a New Stepwise Test without Data Snooping Bias*.
  https://homepage.ntu.edu.tw/~ckuan/pdf/Step-SPA-20090720.pdf

**Datos del repositorio usados en las mediciones**

- `data/seed/sp500_historical_components.csv` — 3.482 snapshots, 1996-01-02 a 2025-08-23, universo
  medio 472,6, 1.128 tickers distintos.
- `data/seed/sp500_constituents.csv` — 503 constituyentes actuales con sector GICS y CIK.
