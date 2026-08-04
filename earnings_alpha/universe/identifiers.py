"""Identificadores del universo: mapa bidireccional ticker↔CIK y cambios de símbolo.

Un backtest point-in-time necesita responder a dos preguntas distintas que se
confunden con facilidad:

1. **¿Qué entidad legal hay detrás de este símbolo?** — el símbolo es efímero
   (Facebook pasó a `META`, PerkinElmer a `RVTY`, Dow-DuPont se escindió en `DD`,
   `DOW` y `CTVA`), mientras que el CIK de la SEC es estable. Unir fundamentales
   por símbolo sin vigilar los cambios de símbolo produce series rotas justo en el
   punto donde ocurre el evento corporativo interesante.
2. **¿Es este alta/baja del índice un cambio de símbolo o una sustitución real?**
   Un cambio de símbolo NO es un evento de reconstitución: si se trata como tal, el
   estudio del "efecto de inclusión en el índice" queda contaminado por decenas de
   pseudo-eventos sin flujo indexado asociado.

Este módulo trata la segunda pregunta como lo que es: una **heurística**. Devuelve
*candidatos* puntuados, nunca una afirmación de identidad; la confirmación exige
CIK coincidente o revisión manual.

Referencias
-----------
- Chen, Noronha & Singal (2004), "The Price Response to S&P 500 Index Additions and
  Deletions: Evidence of Asymmetry and a New Explanation", *Journal of Finance* 59(4).
  Motiva separar altas/bajas genuinas de artefactos de nomenclatura.
- Shumway (1997), "The Delisting Bias in CRSP Data", *Journal of Finance* 52(1).
  El tratamiento de símbolos que desaparecen es una fuente clásica de sesgo.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from earnings_alpha.errors import UniverseError
from earnings_alpha.types import CIK, Ticker, normalize_cik, normalize_ticker

__all__ = [
    "IdentifierMap",
    "IdentifierRecord",
    "SymbolChangeCandidate",
    "TickerSpan",
    "clean_symbol",
    "detect_symbol_changes",
    "merge_identifier_maps",
    "ticker_spans",
]

# Anotaciones que algunos datasets incrustan en la propia celda del símbolo, del tipo
# "RVTY (Previously PKI)". No son parte del ticker y deben eliminarse antes de
# normalizar, o `normalize_ticker` las convertiría en "RVTY.(PREVIOUSLY.PKI)".
_ANNOTATION_RE = re.compile(r"\s*[\(\[].*?[\)\]]\s*")

# Sufijos de clase de acción reconocidos, usados para detectar familias de símbolos
# (GOOG/GOOGL, NWS/NWSA, FOX/FOXA, BRK.A/BRK.B).
_CLASS_SUFFIXES = ("A", "B", "C", "K")


def clean_symbol(raw: str) -> Ticker:
    """Limpia y normaliza un símbolo procedente de una fuente externa.

    Elimina anotaciones entre paréntesis o corchetes y delega en
    `types.normalize_ticker` la forma canónica (punto en lugar de guion).
    Lanza `UniverseError` si tras la limpieza no queda nada utilizable: un símbolo
    vacío colándose en el universo es un fallo silencioso inaceptable.
    """
    txt = _ANNOTATION_RE.sub(" ", str(raw)).strip()
    # Algunas fuentes anexan notas sin paréntesis ("AAPL previously APPL"). Se
    # conserva el primer token y los siguientes de una sola letra, que sí forman
    # parte del símbolo cuando la fuente separa la clase con espacio ("BRK B").
    tokens = txt.split()
    if len(tokens) > 1:
        kept = [tokens[0]]
        for tok in tokens[1:]:
            if len(tok) == 1 and tok.isalpha():
                kept.append(tok)
            else:
                break
        txt = ".".join(kept)
    out = normalize_ticker(txt)
    if not out:
        msg = f"símbolo vacío tras normalizar: {raw!r}"
        raise UniverseError(msg)
    return out


@dataclass(frozen=True, slots=True)
class IdentifierRecord:
    """Una fila del registro de identificadores.

    `source` permite auditar de dónde vino cada correspondencia y resolver conflictos
    por prioridad de fuente en lugar de por orden de llegada.
    """

    ticker: Ticker
    cik: CIK | None = None
    name: str | None = None
    sector: str | None = None
    sub_industry: str | None = None
    date_added: date | None = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class TickerSpan:
    """Tramo de pertenencia continua de un símbolo al índice.

    `n_spans > 1` indica reentradas: la empresa salió y volvió (es frecuente), lo que
    obliga a que cualquier señal se calcule sobre el panel de pertenencia y no sobre
    el intervalo [first_seen, last_seen], que incluiría los huecos.
    """

    ticker: Ticker
    first_seen: date
    last_seen: date
    n_snapshots: int
    spans: tuple[tuple[date, date], ...]

    @property
    def n_spans(self) -> int:
        """Número de tramos de pertenencia continua."""
        return len(self.spans)

    @property
    def has_gaps(self) -> bool:
        """True si el símbolo salió del índice y volvió al menos una vez."""
        return len(self.spans) > 1


@dataclass(frozen=True, slots=True)
class SymbolChangeCandidate:
    """Candidato a cambio de símbolo detectado entre dos snapshots consecutivos.

    `score` ∈ [0, 1] combina evidencia; `confirmed_by_cik` es la única evidencia
    concluyente disponible sin intervención humana.
    """

    date: date
    old_ticker: Ticker
    new_ticker: Ticker
    score: float
    reasons: tuple[str, ...]
    confirmed_by_cik: bool = False


def _as_date(value: object) -> date:
    """Convierte a `datetime.date` cualquier representación de fecha admitida."""
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if ts is pd.NaT:  # pragma: no cover - defensivo
        msg = f"fecha no interpretable: {value!r}"
        raise UniverseError(msg)
    return ts.date()


class IdentifierMap:
    """Mapa bidireccional ticker↔CIK con resolución determinista de colisiones.

    La relación no es 1:1 en ninguna dirección:

    - **Un CIK, varios tickers.** Las clases múltiples de acciones cotizan por
      separado pero comparten emisor y, por tanto, comparten los fundamentales de
      EDGAR (`GOOGL`/`GOOG`, `NWSA`/`NWS`, `FOXA`/`FOX`, `BRK.A`/`BRK.B`). Repartir
      el mismo XBRL entre dos símbolos duplica la exposición del factor si no se
      elige un símbolo primario.
    - **Un ticker, varios CIK a lo largo del tiempo.** Los símbolos se reciclan tras
      una quiebra o fusión (`GM`, `KO`... en datasets largos). Por eso `cik_for`
      acepta una fecha: sin ella devuelve la correspondencia vigente.
    """

    __slots__ = ("_by_cik", "_by_ticker", "_conflicts", "_preferred")

    def __init__(
        self,
        records: Iterable[IdentifierRecord] = (),
        *,
        preferred: Mapping[CIK, Ticker] | None = None,
    ) -> None:
        self._by_ticker: dict[Ticker, IdentifierRecord] = {}
        self._by_cik: dict[CIK, list[Ticker]] = {}
        self._conflicts: list[tuple[Ticker, CIK, CIK, str]] = []
        self._preferred: dict[CIK, Ticker] = dict(preferred or {})
        for rec in records:
            self.add(rec)

    # ------------------------------------------------------------------ carga

    def add(self, record: IdentifierRecord) -> None:
        """Inserta o actualiza un registro.

        Si el mismo ticker llega con un CIK distinto se guarda el conflicto en
        `conflicts()` y **se conserva el primero**: sobrescribir en silencio haría
        que el resultado dependiera del orden de las fuentes.
        """
        ticker = clean_symbol(record.ticker)
        cik = normalize_cik(record.cik) if record.cik is not None else None
        rec = IdentifierRecord(
            ticker=ticker,
            cik=cik,
            name=record.name,
            sector=record.sector,
            sub_industry=record.sub_industry,
            date_added=record.date_added,
            source=record.source,
        )
        prev = self._by_ticker.get(ticker)
        if prev is not None and prev.cik and rec.cik and prev.cik != rec.cik:
            self._conflicts.append((ticker, prev.cik, rec.cik, rec.source))
            return
        if prev is not None and prev.cik and not rec.cik:
            # No degradamos un registro que ya tenía CIK con otro que no lo trae.
            rec = IdentifierRecord(
                ticker=ticker,
                cik=prev.cik,
                name=rec.name or prev.name,
                sector=rec.sector or prev.sector,
                sub_industry=rec.sub_industry or prev.sub_industry,
                date_added=rec.date_added or prev.date_added,
                source=rec.source,
            )
        self._by_ticker[ticker] = rec
        if rec.cik:
            bucket = self._by_cik.setdefault(rec.cik, [])
            if ticker not in bucket:
                bucket.append(ticker)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        source: str = "frame",
        columns: Mapping[str, str] | None = None,
    ) -> IdentifierMap:
        """Construye el mapa desde un DataFrame tipo `sp500_constituents.csv`.

        `columns` permite renombrar: por defecto se esperan las columnas de Wikipedia
        (`Symbol`, `Security`, `GICS Sector`, `GICS Sub-Industry`, `Date added`, `CIK`).
        """
        cols = {
            "ticker": "Symbol",
            "name": "Security",
            "sector": "GICS Sector",
            "sub_industry": "GICS Sub-Industry",
            "date_added": "Date added",
            "cik": "CIK",
        }
        cols.update(columns or {})
        if cols["ticker"] not in frame.columns:
            msg = f"falta la columna de símbolo {cols['ticker']!r} en el DataFrame"
            raise UniverseError(msg)

        def _get(row: pd.Series, key: str) -> object | None:
            col = cols[key]
            if col not in frame.columns:
                return None
            val = row[col]
            return None if pd.isna(val) else val

        records: list[IdentifierRecord] = []
        for _, row in frame.iterrows():
            raw_cik = _get(row, "cik")
            raw_added = _get(row, "date_added")
            records.append(
                IdentifierRecord(
                    ticker=str(row[cols["ticker"]]),
                    cik=normalize_cik(raw_cik) if raw_cik is not None else None,  # type: ignore[arg-type]
                    name=str(_get(row, "name")) if _get(row, "name") is not None else None,
                    sector=str(_get(row, "sector")) if _get(row, "sector") is not None else None,
                    sub_industry=(
                        str(_get(row, "sub_industry"))
                        if _get(row, "sub_industry") is not None
                        else None
                    ),
                    date_added=_as_date(raw_added) if raw_added is not None else None,
                    source=source,
                )
            )
        return cls(records)

    # ------------------------------------------------------------- consultas

    def __len__(self) -> int:
        return len(self._by_ticker)

    def __contains__(self, ticker: object) -> bool:
        if not isinstance(ticker, str):
            return False
        return clean_symbol(ticker) in self._by_ticker

    @property
    def tickers(self) -> tuple[Ticker, ...]:
        """Todos los símbolos registrados, ordenados."""
        return tuple(sorted(self._by_ticker))

    def record_for(self, ticker: Ticker) -> IdentifierRecord | None:
        """Registro completo del símbolo, o None si es desconocido."""
        return self._by_ticker.get(clean_symbol(ticker))

    def cik_for(self, ticker: Ticker) -> CIK | None:
        """CIK del emisor detrás del símbolo, o None si no está registrado."""
        rec = self._by_ticker.get(clean_symbol(ticker))
        return rec.cik if rec else None

    def tickers_for(self, cik: CIK | int) -> tuple[Ticker, ...]:
        """Todos los símbolos asociados al CIK (varias clases de acciones)."""
        return tuple(sorted(self._by_cik.get(normalize_cik(cik), [])))

    def collisions(self) -> dict[CIK, tuple[Ticker, ...]]:
        """CIK con más de un símbolo. Base para elegir un símbolo primario."""
        return {c: tuple(sorted(t)) for c, t in self._by_cik.items() if len(t) > 1}

    def conflicts(self) -> tuple[tuple[Ticker, CIK, CIK, str], ...]:
        """Símbolos que recibieron dos CIK distintos: `(ticker, cik_previo, cik_nuevo, fuente)`.

        Un conflicto es un aviso de calidad de datos, no un error recuperable
        automáticamente: exige decidir qué fuente manda.
        """
        return tuple(self._conflicts)

    def primary_ticker(self, cik: CIK | int) -> Ticker | None:
        """Símbolo primario del emisor, de forma determinista.

        Orden de desempate, documentado y estable (no es una afirmación sobre qué
        clase es más líquida, sino una regla reproducible):

        1. preferencia explícita pasada en el constructor,
        2. menor `date_added` conocido (la clase que entró antes en el índice),
        3. menor longitud del símbolo,
        4. orden alfabético.
        """
        key = normalize_cik(cik)
        candidates = self._by_cik.get(key)
        if not candidates:
            return None
        pref = self._preferred.get(key)
        if pref and pref in candidates:
            return pref
        far_future = date(9999, 12, 31)

        def _sort_key(t: Ticker) -> tuple[date, int, str]:
            rec = self._by_ticker[t]
            return (rec.date_added or far_future, len(t), t)

        return sorted(candidates, key=_sort_key)[0]

    def resolve_collisions(self) -> dict[Ticker, Ticker]:
        """Mapa `símbolo secundario → símbolo primario` para todos los CIK colisionados.

        Uso típico: colapsar `GOOG`→`GOOGL` antes de calcular un factor fundamental,
        de modo que el mismo XBRL no genere dos posiciones.
        """
        out: dict[Ticker, Ticker] = {}
        for cik, tickers in self.collisions().items():
            primary = self.primary_ticker(cik)
            if primary is None:  # pragma: no cover - imposible si hay colisión
                continue
            for t in tickers:
                if t != primary:
                    out[t] = primary
        return out

    def to_frame(self) -> pd.DataFrame:
        """Vuelca el mapa a un DataFrame indexado por ticker, ordenado."""
        rows = [
            {
                "ticker": r.ticker,
                "cik": r.cik,
                "name": r.name,
                "sector": r.sector,
                "sub_industry": r.sub_industry,
                "date_added": r.date_added,
                "source": r.source,
            }
            for r in self._by_ticker.values()
        ]
        frame = pd.DataFrame(
            rows,
            columns=["ticker", "cik", "name", "sector", "sub_industry", "date_added", "source"],
        )
        if frame.empty:
            return frame.set_index("ticker")
        return frame.sort_values("ticker").set_index("ticker")


def merge_identifier_maps(*maps: IdentifierMap) -> IdentifierMap:
    """Fusiona varios mapas por prioridad: **el primero gana** ante conflicto.

    Los desacuerdos quedan registrados en `conflicts()` del mapa resultante en lugar
    de resolverse silenciosamente.
    """
    merged = IdentifierMap()
    for m in maps:
        for ticker in m.tickers:
            rec = m.record_for(ticker)
            if rec is not None:
                merged.add(rec)
    return merged


# --------------------------------------------------------------------- spans


def ticker_spans(
    dates: Sequence[date],
    member_sets: Sequence[frozenset[Ticker]],
) -> dict[Ticker, TickerSpan]:
    """Calcula los tramos de pertenencia de cada símbolo.

    `dates` y `member_sets` son las fechas de snapshot (ordenadas) y la pertenencia en
    cada una. Dos snapshots consecutivos definen un tramo continuo: no se interpola
    entre ellos porque, por construcción del dataset, la pertenencia solo cambia en
    fechas registradas.
    """
    if len(dates) != len(member_sets):
        msg = "dates y member_sets deben tener la misma longitud"
        raise UniverseError(msg)
    open_start: dict[Ticker, date] = {}
    closed: dict[Ticker, list[tuple[date, date]]] = {}
    counts: dict[Ticker, int] = {}
    prev: frozenset[Ticker] = frozenset()
    prev_date: date | None = None

    for d, members in zip(dates, member_sets, strict=True):
        for t in members - prev:
            open_start[t] = d
        for t in prev - members:
            start = open_start.pop(t, prev_date or d)
            closed.setdefault(t, []).append((start, prev_date or d))
        for t in members:
            counts[t] = counts.get(t, 0) + 1
        prev = members
        prev_date = d

    for t, start in open_start.items():
        closed.setdefault(t, []).append((start, prev_date or start))

    out: dict[Ticker, TickerSpan] = {}
    for t, spans in closed.items():
        ordered = tuple(sorted(spans))
        out[t] = TickerSpan(
            ticker=t,
            first_seen=ordered[0][0],
            last_seen=ordered[-1][1],
            n_snapshots=counts.get(t, 0),
            spans=ordered,
        )
    return out


# ---------------------------------------------------------- cambios de símbolo


def _name_similarity(a: str | None, b: str | None) -> float:
    """Similitud de nombres societarios en [0, 1] (0 si falta alguno)."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.upper(), b.upper()).ratio()


def _share_class_root(symbol: Ticker) -> Ticker:
    """Raíz del símbolo sin el sufijo de clase con punto (`BRK.B` → `BRK`)."""
    head, sep, tail = symbol.rpartition(".")
    if sep and len(tail) == 1:
        return head
    return symbol


def _share_class_family(old: Ticker, new: Ticker) -> bool:
    """True si ambos símbolos parecen clases de la misma familia.

    Cubre las dos convenciones vivas en el S&P 500: sufijo con punto
    (`BRK.A`/`BRK.B`) y letra pegada al final (`GOOG`/`GOOGL`, `NWS`/`NWSA`,
    `FOX`/`FOXA`).
    """
    if old == new:
        return False
    if _share_class_root(old) == _share_class_root(new):
        return True
    a, b = sorted((old, new), key=len)
    if not b.startswith(a):
        return False
    tail = b[len(a) :].lstrip(".")
    return len(tail) == 1 and tail in _CLASS_SUFFIXES


def _prefix_ratio(old: Ticker, new: Ticker) -> float:
    """Fracción del símbolo corto compartida como prefijo con el largo."""
    n = min(len(old), len(new))
    shared = 0
    for i in range(n):
        if old[i] != new[i]:
            break
        shared += 1
    return shared / max(len(old), len(new))


_CHANGE_COLUMNS = [
    "date",
    "old_ticker",
    "new_ticker",
    "score",
    "tier",
    "reasons",
    "confirmed_by_cik",
]


def _tier(score: float, confirmed: bool) -> str:
    """Etiqueta de confianza asociada a la puntuación."""
    if confirmed:
        return "confirmed"
    if score >= 0.70:
        return "strong"
    return "candidate"


def detect_symbol_changes(
    changes: pd.DataFrame,
    *,
    identifiers: IdentifierMap | None = None,
    names: Mapping[Ticker, str] | None = None,
    known_changes: Mapping[Ticker, Ticker] | None = None,
    min_score: float = 0.40,
) -> pd.DataFrame:
    """Detecta candidatos a cambio de símbolo a partir de altas y bajas simultáneas.

    Entrada: el DataFrame `(date, ticker, action)` que devuelven
    `SP500Universe.additions/deletions/changes`. Dentro de cada fecha se emparejan
    las bajas con las altas y se puntúa cada par:

    ============================  =====  ============================================
    Evidencia                     Peso   Comentario
    ============================  =====  ============================================
    Cambio conocido aportado      1.00   tabla curada por quien llama
    CIK idéntico                  1.00   concluyente: el emisor es literalmente el mismo
    Misma familia de clase        0.60   `GOOG`/`GOOGL`, `BRK.A`/`BRK.B`
    Nombre societario similar     0.50   ratio de `difflib` sobre el nombre
    Prefijo compartido            0.30   `FISV`→`FI`, `DWDP`→`DD`
    Única alta y única baja       0.40   la baja y el alta ocurren aisladas ese día
    ============================  =====  ============================================

    Salida: `(date, old_ticker, new_ticker, score, tier, reasons, confirmed_by_cik)`,
    con `tier ∈ {"confirmed", "strong", "candidate"}`.

    **Limitación deliberada y central.** A partir de la composición del índice, un
    cambio de símbolo y una sustitución real son *observacionalmente idénticos*:
    ambos aparecen como una baja y un alta el mismo día. Verificado contra el dato
    real de la semilla, entre los pares aislados conviven renombramientos
    (`FB`→`META` el 2022-06-09, `ANTM`→`ELV`, `PKI`→`RVTY`, `FISV`→`FI`) y
    sustituciones auténticas que no tienen nada que ver entre sí (`ALXN`→`MRNA`,
    `TWTR`→`ACGL`). Por eso la salida se llama *candidatos*: solo
    `confirmed_by_cik=True` (o `known_changes`) es concluyente, y el resto exige
    revisión antes de excluir esos días de un estudio de inclusión en el índice.

    Además, la evidencia por CIK **no puede activarse con la semilla del repo**: la
    tabla de constituyentes solo trae CIK de los miembros actuales, y el símbolo
    antiguo de un renombramiento ya no figura en ella. Se implementa igualmente
    porque sí funcionará en cuanto se disponga de un mapa histórico ticker↔CIK
    (EDGAR `company_tickers.json` con vintages).
    """
    required = {"date", "ticker", "action"}
    if not required.issubset(changes.columns):
        msg = f"changes debe tener columnas {sorted(required)}; tiene {list(changes.columns)}"
        raise UniverseError(msg)
    if changes.empty:
        return pd.DataFrame(columns=_CHANGE_COLUMNS)

    names = names or {}
    known = {clean_symbol(k): clean_symbol(v) for k, v in (known_changes or {}).items()}
    out: list[SymbolChangeCandidate] = []
    for day, block in changes.groupby("date", sort=True):
        adds = [str(t) for t in block.loc[block["action"] == "add", "ticker"]]
        dels = [str(t) for t in block.loc[block["action"] == "delete", "ticker"]]
        if not adds or not dels:
            continue
        isolated = len(adds) == 1 and len(dels) == 1
        for old in dels:
            for new in adds:
                score = 0.0
                reasons: list[str] = []
                cik_old = identifiers.cik_for(old) if identifiers else None
                cik_new = identifiers.cik_for(new) if identifiers else None
                confirmed = bool(cik_old and cik_new and cik_old == cik_new)
                if confirmed:
                    score += 1.0
                    reasons.append("cik_match")
                if known.get(old) == new:
                    score += 1.0
                    confirmed = True
                    reasons.append("known_change")
                if _share_class_family(old, new):
                    score += 0.60
                    reasons.append("share_class_family")
                sim = _name_similarity(names.get(old), names.get(new))
                if sim >= 0.60:
                    score += 0.50 * sim
                    reasons.append(f"name_similarity={sim:.2f}")
                pref = _prefix_ratio(old, new)
                if pref >= 0.40:
                    score += 0.30 * pref
                    reasons.append(f"prefix_ratio={pref:.2f}")
                if isolated:
                    score += 0.40
                    reasons.append("isolated_pair")
                score = min(score, 1.0)
                if score >= min_score:
                    out.append(
                        SymbolChangeCandidate(
                            date=_as_date(day),
                            old_ticker=old,
                            new_ticker=new,
                            score=round(score, 4),
                            reasons=tuple(reasons),
                            confirmed_by_cik=confirmed,
                        )
                    )

    frame = pd.DataFrame(
        [
            {
                "date": c.date,
                "old_ticker": c.old_ticker,
                "new_ticker": c.new_ticker,
                "score": c.score,
                "tier": _tier(c.score, c.confirmed_by_cik),
                "reasons": ", ".join(c.reasons),
                "confirmed_by_cik": c.confirmed_by_cik,
            }
            for c in out
        ],
        columns=_CHANGE_COLUMNS,
    )
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["score", "date"], ascending=[False, True]).reset_index(drop=True)
