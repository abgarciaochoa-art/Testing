"""Pertenencia point-in-time al S&P 500 y detección de altas/bajas del índice.

El universo es la primera decisión de cualquier backtest cross-section y la más
fácil de equivocar. Usar la lista *de hoy* para estudiar 2008 produce sesgo de
supervivencia: las empresas que quebraron (Lehman, Circuit City, Washington Mutual)
desaparecen de la muestra justo cuando su retorno era catastrófico, y el estudio
mide una cartera que nadie pudo comprar.

Este módulo resuelve la pertenencia leyendo la composición **diaria real** de
`data/seed/sp500_historical_components.csv` (1996-01-02 → 2025-08-23; 3.482 filas que
colapsan a 3.480 fechas únicas, y 1.125 símbolos distintos a lo largo de la historia)
y aplicando una regla de resolución temporal estricta: para una fecha sin snapshot se
usa el snapshot **anterior** más próximo (forward-fill), nunca el posterior, porque
el posterior no era conocible.

Además expone las **altas y bajas** del índice por diferencia entre snapshots
consecutivos. Ese es el insumo para estudiar el efecto de inclusión, uno de los
resultados más replicados de la literatura de microestructura:

- Harris & Gurel (1986), "Price and Volume Effects Associated with Changes in the
  S&P 500 List", *Journal of Finance* 41(4): efecto precio transitorio con reversión.
- Shleifer (1986), "Do Demand Curves for Stocks Slope Down?", *Journal of Finance*
  41(3): efecto permanente, evidencia contra la sustituibilidad perfecta.
- Chen, Noronha & Singal (2004), *Journal of Finance* 59(4): asimetría entre altas y
  bajas, y la importancia de fechar correctamente anuncio vs. efectividad.

Advertencia metodológica: el dataset semilla registra la composición **efectiva**.
El anuncio de S&P Dow Jones Indices precede a la efectividad en varios días
hábiles, y buena parte del retorno anormal ocurre entre ambos. Un estudio del efecto
de inclusión debe fechar por anuncio; con este fichero solo se dispone de la fecha
efectiva, lo que se anota en `docs/OPEN_QUESTIONS.md`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd

from earnings_alpha.config import Settings, get_settings
from earnings_alpha.errors import (
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
    UniverseError,
)
from earnings_alpha.types import CIK, Ticker
from earnings_alpha.universe.identifiers import (
    IdentifierMap,
    TickerSpan,
    clean_symbol,
    detect_symbol_changes,
    ticker_spans,
)
from earnings_alpha.universe.sources import (
    RefreshReport,
    UniverseSource,
    build_source,
    history_frame_to_records,
    read_history_csv,
    refresh_history_file,
)

__all__ = [
    "DateLike",
    "HistorySnapshotIndex",
    "SP500Universe",
    "UniverseProvider",
]

DateLike: TypeAlias = dt.date | dt.datetime | str | pd.Timestamp
"""Cualquier representación de fecha aceptada por la API pública del módulo."""

HISTORY_FILENAME = "sp500_historical_components.csv"
CONSTITUENTS_FILENAME = "sp500_constituents.csv"


@runtime_checkable
class UniverseProvider(Protocol):
    """Contrato de pertenencia al índice (sección 3.1 de `docs/ARCHITECTURE.md`)."""

    def members_on(self, d: dt.date) -> list[Ticker]:
        """Símbolos que pertenecían al índice en la fecha `d`."""
        ...

    def membership_panel(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """DataFrame booleano index=date, columns=ticker. True = en el índice ese día."""
        ...

    def cik_for(self, t: Ticker, on: dt.date | None = None) -> CIK | None:
        """CIK del emisor detrás del símbolo."""
        ...

    def sector_for(self, t: Ticker) -> str | None:
        """Sector GICS del símbolo."""
        ...


def _as_timestamp(value: DateLike) -> pd.Timestamp:
    """Normaliza a `pd.Timestamp` tz-naive a medianoche, según la convención del repo."""
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        msg = f"fecha no interpretable: {value!r}"
        raise UniverseError(msg) from exc
    if ts is pd.NaT or pd.isna(ts):
        msg = f"fecha no interpretable: {value!r}"
        raise UniverseError(msg)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


# ------------------------------------------------------- índice de snapshots


@dataclass(frozen=True, slots=True)
class HistorySnapshotIndex:
    """Representación vectorizada del histórico de composición, lista para consultar.

    La matriz booleana `matrix[i, j]` (snapshot i x símbolo j) es lo que hace que
    `membership_panel`, `additions` y `deletions` sean operaciones de numpy y no
    bucles sobre 3.480 conjuntos de Python. Con 3.480 x 1.125 ocupa ~3,9 MB, así que
    se construye una sola vez (~0,8 s) y se cachea por (ruta, mtime, tamaño); a partir
    de ahí el panel de historia completa se materializa en ~10 ms.
    """

    dates: np.ndarray
    """`datetime64[ns]` ordenado y sin duplicados: fechas de snapshot."""
    tickers: tuple[Ticker, ...]
    """Todos los símbolos vistos alguna vez, ordenados alfabéticamente."""
    col_index: Mapping[Ticker, int]
    matrix: np.ndarray
    """Booleana `(n_snapshots, n_tickers)`."""
    source_path: Path

    @property
    def n_snapshots(self) -> int:
        """Número de snapshots registrados."""
        return int(self.dates.shape[0])

    @property
    def first_date(self) -> pd.Timestamp:
        """Fecha del primer snapshot disponible."""
        return pd.Timestamp(self.dates[0])

    @property
    def last_date(self) -> pd.Timestamp:
        """Fecha del último snapshot disponible."""
        return pd.Timestamp(self.dates[-1])

    def position(self, ts: pd.Timestamp) -> int:
        """Índice del snapshot vigente en `ts` (forward-fill), o -1 si es anterior a todo.

        `searchsorted(..., side="right") - 1` implementa exactamente la regla
        point-in-time: se toma el último snapshot cuya fecha es **≤** `ts`.
        """
        return int(np.searchsorted(self.dates, np.datetime64(ts), side="right")) - 1

    def members_at_position(self, pos: int) -> list[Ticker]:
        """Símbolos activos en el snapshot `pos`, ordenados."""
        cols = np.flatnonzero(self.matrix[pos])
        return [self.tickers[j] for j in cols]

    def member_sets(self) -> list[frozenset[Ticker]]:
        """Composición de cada snapshot como conjuntos (para análisis de tramos)."""
        return [frozenset(self.members_at_position(i)) for i in range(self.n_snapshots)]


def _build_index(path: Path) -> HistorySnapshotIndex:
    """Parsea el CSV de composición y construye la matriz booleana."""
    frame = read_history_csv(path)
    records = history_frame_to_records(frame)
    if not records:
        msg = f"{path}: no hay snapshots que cargar"
        raise DataQualityError(msg)

    dates = np.array([np.datetime64(d) for d, _ in records], dtype="datetime64[ns]")
    all_tickers = sorted({t for _, members in records for t in members})
    col_index = {t: j for j, t in enumerate(all_tickers)}

    rows: list[int] = []
    cols: list[int] = []
    for i, (_, members) in enumerate(records):
        for t in members:
            rows.append(i)
            cols.append(col_index[t])
    matrix = np.zeros((len(records), len(all_tickers)), dtype=bool)
    matrix[np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)] = True

    return HistorySnapshotIndex(
        dates=dates,
        tickers=tuple(all_tickers),
        col_index=col_index,
        matrix=matrix,
        source_path=path,
    )


@lru_cache(maxsize=8)
def _cached_index(path_str: str, mtime_ns: int, size: int) -> HistorySnapshotIndex:
    """Caché del parseo, invalidada automáticamente por mtime y tamaño del fichero.

    Incluir `mtime_ns` y `size` en la clave es lo que permite que `refresh()` sea
    visible de inmediato sin exponer un `cache_clear()` que alguien olvidaría llamar.
    """
    del mtime_ns, size  # solo forman parte de la clave de caché
    return _build_index(Path(path_str))


@lru_cache(maxsize=8)
def _cached_constituents(path_str: str, mtime_ns: int, size: int) -> pd.DataFrame:
    """Caché de la tabla de constituyentes vigentes (metadatos GICS y CIK)."""
    del mtime_ns, size
    path = Path(path_str)
    if not path.exists():
        raise ProviderUnavailable("sp500_seed_constituents", f"no existe el fichero {path}")
    frame = pd.read_csv(path)
    if "Symbol" not in frame.columns:
        msg = f"{path}: falta la columna 'Symbol'"
        raise DataQualityError(msg)
    frame = frame.copy()
    frame["ticker"] = [clean_symbol(str(s)) for s in frame["Symbol"]]
    if frame["ticker"].duplicated().any():
        dups = sorted(frame.loc[frame["ticker"].duplicated(), "ticker"])
        msg = f"{path}: símbolos duplicados tras normalizar: {dups}"
        raise DataQualityError(msg)
    return frame.set_index("ticker").sort_index()


def _stat_key(path: Path) -> tuple[str, int, int]:
    """Clave de caché de un fichero: ruta, mtime en ns y tamaño."""
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


# ------------------------------------------------------------------ universo


class SP500Universe:
    """Universo S&P 500 point-in-time sobre los ficheros semilla del repo.

    Implementa `UniverseProvider`. Todas las consultas por fecha aplican forward-fill
    desde el snapshot anterior; ninguna mira hacia adelante.

    Ejemplo
    -------
    >>> u = SP500Universe()                                   # doctest: +SKIP
    >>> "LEH" in u.members_on(dt.date(2008, 6, 30))           # doctest: +SKIP
    True
    >>> "TSLA" in u.members_on(dt.date(2008, 6, 30))          # doctest: +SKIP
    False
    """

    def __init__(
        self,
        history_path: Path | str | None = None,
        constituents_path: Path | str | None = None,
        *,
        settings: Settings | None = None,
        max_forward_fill_days: int | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._history_path = Path(history_path or cfg.seed_dir / HISTORY_FILENAME)
        self._constituents_path = Path(constituents_path or cfg.seed_dir / CONSTITUENTS_FILENAME)
        if max_forward_fill_days is not None and max_forward_fill_days < 0:
            msg = "max_forward_fill_days debe ser >= 0"
            raise UniverseError(msg)
        self._max_ffill = max_forward_fill_days
        self._identifiers: IdentifierMap | None = None

    # ------------------------------------------------------------- estado

    @property
    def history_path(self) -> Path:
        """Ruta del CSV de composición histórica en uso."""
        return self._history_path

    @property
    def constituents_path(self) -> Path:
        """Ruta del CSV de constituyentes vigentes en uso."""
        return self._constituents_path

    @property
    def index(self) -> HistorySnapshotIndex:
        """Histórico parseado y cacheado (matriz booleana snapshot x ticker)."""
        if not self._history_path.exists():
            raise ProviderUnavailable(
                "sp500_seed_history", f"no existe el fichero {self._history_path}"
            )
        return _cached_index(*_stat_key(self._history_path))

    @property
    def constituents(self) -> pd.DataFrame:
        """Tabla de constituyentes vigentes, indexada por ticker normalizado."""
        if not self._constituents_path.exists():
            raise ProviderUnavailable(
                "sp500_seed_constituents", f"no existe el fichero {self._constituents_path}"
            )
        return _cached_constituents(*_stat_key(self._constituents_path))

    @property
    def first_snapshot(self) -> dt.date:
        """Primera fecha con composición registrada."""
        return self.index.first_date.date()

    @property
    def last_snapshot(self) -> dt.date:
        """Última fecha con composición registrada."""
        return self.index.last_date.date()

    def snapshot_dates(self, start: DateLike | None = None, end: DateLike | None = None
                       ) -> pd.DatetimeIndex:
        """Fechas de snapshot dentro del rango (ambos extremos incluidos)."""
        idx = pd.DatetimeIndex(self.index.dates, name="date")
        if start is not None:
            idx = idx[idx >= _as_timestamp(start)]
        if end is not None:
            idx = idx[idx <= _as_timestamp(end)]
        return idx

    @property
    def all_tickers(self) -> tuple[Ticker, ...]:
        """Todos los símbolos que pertenecieron al índice en algún momento."""
        return self.index.tickers

    # --------------------------------------------------- pertenencia por fecha

    def _position(self, d: DateLike) -> int:
        """Posición del snapshot vigente en `d`, validando la historia disponible."""
        idx = self.index
        ts = _as_timestamp(d)
        pos = idx.position(ts)
        if pos < 0:
            msg = (
                f"no hay composición registrada para {ts.date()}: el primer snapshot es "
                f"{idx.first_date.date()}. Usar un universo posterior o ampliar el histórico."
            )
            raise InsufficientHistory(msg)
        if self._max_ffill is not None:
            gap = (ts - pd.Timestamp(idx.dates[pos])).days
            if gap > self._max_ffill:
                msg = (
                    f"el snapshot vigente para {ts.date()} es de "
                    f"{pd.Timestamp(idx.dates[pos]).date()} ({gap} días antes), por encima del "
                    f"máximo permitido de {self._max_ffill}"
                )
                raise InsufficientHistory(msg)
        return pos

    def members_on(self, d: DateLike) -> list[Ticker]:
        """Composición del índice vigente en `d`, ordenada alfabéticamente.

        Regla point-in-time: si `d` no es fecha de snapshot se devuelve la composición
        del snapshot **anterior** más próximo. Nunca la del posterior, que no era
        conocible en `d`. Antes del primer snapshot lanza `InsufficientHistory`.
        """
        return self.index.members_at_position(self._position(d))

    def members_set_on(self, d: DateLike) -> frozenset[Ticker]:
        """Igual que `members_on` pero como conjunto, para pertenencias masivas."""
        return frozenset(self.members_on(d))

    def is_member(self, t: Ticker, d: DateLike) -> bool:
        """True si el símbolo pertenecía al índice en `d`. O(1) tras cargar el índice."""
        idx = self.index
        col = idx.col_index.get(clean_symbol(t))
        if col is None:
            return False
        return bool(idx.matrix[self._position(d), col])

    def n_members_on(self, d: DateLike) -> int:
        """Número de miembros vigentes en `d`."""
        return int(self.index.matrix[self._position(d)].sum())

    def snapshot_date_for(self, d: DateLike) -> dt.date:
        """Fecha del snapshot que se usa para responder por `d` (auditoría del ffill)."""
        return pd.Timestamp(self.index.dates[self._position(d)]).date()

    # ------------------------------------------------------------- paneles

    def membership_panel(
        self,
        start: DateLike,
        end: DateLike,
        *,
        freq: str | None = None,
        tickers: Sequence[Ticker] | None = None,
    ) -> pd.DataFrame:
        """Panel booleano de pertenencia, `index=date`, `columns=ticker`.

        Por defecto el índice contiene **solo fechas relevantes**: la fecha `start`
        (con la composición vigente en ella, aunque no haya snapshot ese día) más cada
        fecha de snapshot dentro de `(start, end]`. Es decir, una fila por cambio: con
        3.482 snapshots x ~1.126 símbolos históricos el panel es una vista de la
        matriz cacheada y no reconstruye nada.

        `freq` densifica el panel a una malla temporal regular con forward-fill —
        `"D"` para días naturales, `"B"` para días hábiles. Úsalo solo si el consumidor
        necesita alineación diaria, porque multiplica el tamaño por ~250/año.

        `tickers` restringe las columnas; los símbolos desconocidos aparecen como
        columna íntegramente False en vez de desaparecer, para que el consumidor
        detecte que pidió algo que nunca estuvo en el índice.
        """
        idx = self.index
        ts_start, ts_end = _as_timestamp(start), _as_timestamp(end)
        if ts_start > ts_end:
            msg = f"rango invertido: start={ts_start.date()} > end={ts_end.date()}"
            raise UniverseError(msg)
        pos_start = self._position(ts_start)
        pos_end = self._position(ts_end)

        positions = [pos_start, *range(pos_start + 1, pos_end + 1)]
        row_dates = [ts_start, *(pd.Timestamp(idx.dates[p]) for p in positions[1:])]
        sub = idx.matrix[np.asarray(positions, dtype=np.int64)]

        if tickers is None:
            keep = np.flatnonzero(sub.any(axis=0))
            cols = [idx.tickers[j] for j in keep]
            data = sub[:, keep]
        else:
            wanted = [clean_symbol(t) for t in tickers]
            data = np.zeros((sub.shape[0], len(wanted)), dtype=bool)
            for k, t in enumerate(wanted):
                col = idx.col_index.get(t)
                if col is not None:
                    data[:, k] = sub[:, col]
            cols = wanted

        panel = pd.DataFrame(
            data,
            index=pd.DatetimeIndex(row_dates, name="date"),
            columns=pd.Index(cols, name="ticker"),
        )
        if freq is None:
            return panel
        grid = pd.date_range(ts_start, ts_end, freq=freq, name="date")
        if len(grid) == 0:
            return panel.iloc[:0]
        return panel.reindex(panel.index.union(grid)).ffill().reindex(grid).astype(bool)

    def membership_index(self, start: DateLike, end: DateLike, *, freq: str = "B") -> pd.MultiIndex:
        """MultiIndex canónico `(date, ticker)` de los pares vigentes en el rango.

        Es el índice sobre el que el resto de la plataforma construye sus paneles
        (sección 1 del contrato): garantiza que ninguna señal se calcule para un par
        fecha-símbolo que no estaba en el índice ese día.
        """
        panel = self.membership_panel(start, end, freq=freq)
        rows, cols = np.nonzero(panel.to_numpy())
        return pd.MultiIndex.from_arrays(
            [panel.index.to_numpy()[rows], panel.columns.to_numpy()[cols]],
            names=["date", "ticker"],
        )

    def member_counts(
        self, start: DateLike | None = None, end: DateLike | None = None
    ) -> pd.Series:
        """Número de miembros en cada fecha de snapshot del rango.

        Diagnóstico rápido de integridad: una caída brusca del recuento delata un
        snapshot truncado por un fallo de descarga, no una reconstitución.
        """
        idx = self.index
        counts = idx.matrix.sum(axis=1)
        series = pd.Series(
            counts, index=pd.DatetimeIndex(idx.dates, name="date"), name="n_members"
        )
        if start is not None:
            series = series[series.index >= _as_timestamp(start)]
        if end is not None:
            series = series[series.index <= _as_timestamp(end)]
        return series

    # ----------------------------------------------------------- altas/bajas

    def changes(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        *,
        action: Literal["add", "delete", "both"] = "both",
    ) -> pd.DataFrame:
        """Altas y bajas del índice, por diferencia entre snapshots consecutivos.

        Devuelve `(date, ticker, action)` con `action ∈ {"add", "delete"}`, ordenado
        por fecha y símbolo. `date` es la fecha del snapshot en que el cambio ya es
        visible, es decir la **fecha efectiva** según el dataset.

        El primer snapshot del histórico no genera altas: sus miembros no "entraron"
        ese día, simplemente es donde empieza la observación. Confundir ambas cosas
        crearía ~470 pseudo-eventos de inclusión el 1996-01-02.

        Con `start` anterior al primer snapshot lanza `InsufficientHistory`: no se
        puede afirmar qué cambió en un periodo que no se observó.
        """
        idx = self.index
        ts_start = _as_timestamp(start) if start is not None else idx.first_date
        ts_end = _as_timestamp(end) if end is not None else idx.last_date
        if ts_start > ts_end:
            msg = f"rango invertido: start={ts_start.date()} > end={ts_end.date()}"
            raise UniverseError(msg)
        if ts_start < idx.first_date:
            msg = (
                f"no se pueden derivar cambios desde {ts_start.date()}: la observación "
                f"empieza en {idx.first_date.date()}"
            )
            raise InsufficientHistory(msg)

        lo = int(np.searchsorted(idx.dates, np.datetime64(ts_start), side="left"))
        hi = int(np.searchsorted(idx.dates, np.datetime64(ts_end), side="right"))
        lo = max(lo, 1)  # el snapshot 0 no tiene predecesor observable

        rows: list[dict[str, object]] = []
        for i in range(lo, hi):
            day = pd.Timestamp(idx.dates[i])
            cur, prev = idx.matrix[i], idx.matrix[i - 1]
            if action in ("add", "both"):
                for j in np.flatnonzero(cur & ~prev):
                    rows.append({"date": day, "ticker": idx.tickers[j], "action": "add"})
            if action in ("delete", "both"):
                for j in np.flatnonzero(prev & ~cur):
                    rows.append({"date": day, "ticker": idx.tickers[j], "action": "delete"})

        frame = pd.DataFrame(rows, columns=["date", "ticker", "action"])
        if frame.empty:
            frame["date"] = pd.Series(dtype="datetime64[ns]")
            return frame
        return frame.sort_values(["date", "action", "ticker"]).reset_index(drop=True)

    def additions(self, start: DateLike | None = None, end: DateLike | None = None) -> pd.DataFrame:
        """Altas del índice en el rango: `(date, ticker, action="add")`."""
        return self.changes(start, end, action="add")

    def deletions(self, start: DateLike | None = None, end: DateLike | None = None) -> pd.DataFrame:
        """Bajas del índice en el rango: `(date, ticker, action="delete")`."""
        return self.changes(start, end, action="delete")

    def turnover(self, start: DateLike | None = None, end: DateLike | None = None) -> pd.Series:
        """Rotación anual del índice: número de altas por año natural.

        Referencia útil para contrastar: S&P 500 rota históricamente ~20-25 nombres al
        año. Un año con 60 altas en este dataset apunta a un artefacto de datos.
        """
        adds = self.additions(start, end)
        if adds.empty:
            return pd.Series(dtype="int64", name="additions")
        return (
            adds.groupby(adds["date"].dt.year)["ticker"].count().rename("additions").sort_index()
        )

    # --------------------------------------------------------- identificadores

    def identifiers(self) -> IdentifierMap:
        """Mapa ticker↔CIK construido desde la tabla de constituyentes vigentes.

        **Limitación honesta**: la semilla solo trae CIK de los miembros *actuales*,
        de modo que los ~620 símbolos históricos ya desaparecidos no tienen CIK. Se
        expone tal cual (`cik_for` devuelve None) en lugar de inventar una
        correspondencia; `coverage()` cuantifica el hueco.
        """
        if self._identifiers is None:
            self._identifiers = IdentifierMap.from_frame(
                self.constituents.reset_index(drop=True), source="sp500_constituents_seed"
            )
        return self._identifiers

    def cik_for(self, t: Ticker, on: DateLike | None = None) -> CIK | None:
        """CIK del emisor detrás del símbolo, o None si no está registrado.

        `on` forma parte del contrato para permitir vintages cuando exista un mapa
        histórico de identificadores. Hoy la semilla no tiene vintage: si se pasa
        `on`, se **valida** que el símbolo estuviera en el índice ese día y se
        devuelve `None` en caso contrario, que es la respuesta conservadora — mejor un
        hueco explícito que un CIK atribuido a una fecha en la que ese símbolo podía
        pertenecer a otra empresa (los símbolos se reciclan tras quiebras y fusiones).
        """
        ticker = clean_symbol(t)
        if on is not None and not self.is_member(ticker, on):
            return None
        return self.identifiers().cik_for(ticker)

    def tickers_for_cik(self, cik: CIK | int) -> tuple[Ticker, ...]:
        """Símbolos asociados a un CIK (clases múltiples de acciones)."""
        return self.identifiers().tickers_for(cik)

    def sector_for(self, t: Ticker) -> str | None:
        """Sector GICS del símbolo, o None si no es constituyente actual."""
        return self._meta(t, "GICS Sector")

    def sub_industry_for(self, t: Ticker) -> str | None:
        """Sub-industria GICS del símbolo, o None si no es constituyente actual."""
        return self._meta(t, "GICS Sub-Industry")

    def security_name_for(self, t: Ticker) -> str | None:
        """Denominación de la compañía, o None si no es constituyente actual."""
        return self._meta(t, "Security")

    def date_added_for(self, t: Ticker) -> dt.date | None:
        """Fecha de alta en el índice según la tabla de constituyentes vigentes.

        Es la fecha de la *última* incorporación que registra Wikipedia; para altas y
        bajas derivadas de la composición real usar `additions()`/`deletions()`.
        """
        raw = self._meta(t, "Date added")
        if raw is None:
            return None
        ts = pd.to_datetime(str(raw)[:10], errors="coerce")
        return None if ts is pd.NaT or pd.isna(ts) else ts.date()

    def _meta(self, t: Ticker, column: str) -> str | None:
        """Lee un metadato de la tabla de constituyentes con normalización de símbolo."""
        frame = self.constituents
        if column not in frame.columns:
            msg = f"{self._constituents_path}: falta la columna {column!r}"
            raise DataQualityError(msg)
        ticker = clean_symbol(t)
        if ticker not in frame.index:
            return None
        val = frame[column].to_numpy()[frame.index.get_loc(ticker)]
        if pd.isna(val):
            return None
        return str(val)

    def sector_map(self, d: DateLike | None = None) -> pd.Series:
        """Serie `ticker → sector GICS` para los miembros vigentes en `d`.

        Los miembros sin sector conocido (símbolos históricos ya desaparecidos)
        aparecen con `NaN`, nunca se les asigna un sector por defecto: neutralizar por
        un sector inventado desplaza la exposición del factor de forma silenciosa.
        """
        members = self.members_on(d) if d is not None else list(self.constituents.index)
        sectors = {t: self.sector_for(t) for t in members}
        return pd.Series(sectors, name="sector").rename_axis("ticker").sort_index()

    def members_by_sector(self, d: DateLike) -> dict[str, list[Ticker]]:
        """Miembros vigentes en `d` agrupados por sector GICS (`"UNKNOWN"` si falta)."""
        out: dict[str, list[Ticker]] = {}
        for t in self.members_on(d):
            out.setdefault(self.sector_for(t) or "UNKNOWN", []).append(t)
        return {k: sorted(v) for k, v in sorted(out.items())}

    # ------------------------------------------------------------ diagnóstico

    def spans(self) -> dict[Ticker, TickerSpan]:
        """Tramos de pertenencia continua de cada símbolo (detecta reentradas)."""
        idx = self.index
        dates = [pd.Timestamp(d).date() for d in idx.dates]
        return ticker_spans(dates, idx.member_sets())

    def symbol_change_candidates(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        *,
        min_score: float = 0.40,
        known_changes: Mapping[Ticker, Ticker] | None = None,
    ) -> pd.DataFrame:
        """Candidatos a cambio de símbolo entre las altas y bajas del rango.

        Se apoya en `identifiers.detect_symbol_changes`; el resultado es heurístico y
        debe revisarse antes de excluir esos días de un estudio de inclusión.
        `known_changes` permite inyectar una tabla curada de renombramientos ya
        verificados, que se marcan como concluyentes.
        """
        names = {t: (self.security_name_for(t) or "") for t in self.all_tickers}
        return detect_symbol_changes(
            self.changes(start, end),
            identifiers=self.identifiers(),
            names={k: v for k, v in names.items() if v},
            known_changes=known_changes,
            min_score=min_score,
        )

    def coverage(self) -> dict[str, object]:
        """Métricas de cobertura de la semilla, para saber de qué NO se dispone."""
        idx = self.index
        ident = self.identifiers()
        historic = set(idx.tickers)
        current = set(self.constituents.index)
        with_cik = {t for t in historic if ident.cik_for(t)}
        counts = idx.matrix.sum(axis=1)
        return {
            "n_snapshots": idx.n_snapshots,
            "first_date": idx.first_date.date(),
            "last_date": idx.last_date.date(),
            "n_tickers_historic": len(historic),
            "n_tickers_current": len(current),
            "n_with_cik": len(with_cik),
            "cik_coverage_historic": round(len(with_cik) / max(len(historic), 1), 4),
            "min_members": int(counts.min()),
            "max_members": int(counts.max()),
            "median_members": float(np.median(counts)),
            "n_cik_collisions": len(ident.collisions()),
        }

    def validate(self) -> list[str]:
        """Comprobaciones de integridad del histórico; devuelve la lista de avisos.

        No lanza: está pensada para ejecutarse en CI y en el informe de refresco, donde
        interesa ver *todos* los problemas de una vez y no solo el primero.
        """
        idx = self.index
        issues: list[str] = []
        dates = pd.DatetimeIndex(idx.dates)
        if not dates.is_monotonic_increasing:
            issues.append("las fechas de snapshot no son crecientes")
        if dates.has_duplicates:
            issues.append("hay fechas de snapshot duplicadas tras normalizar")
        counts = idx.matrix.sum(axis=1)
        weird = dates[(counts < 400) | (counts > 520)]
        if len(weird):
            issues.append(
                f"{len(weird)} snapshots con recuento implausible "
                f"(primero: {weird[0].date()}, {int(counts[(counts < 400) | (counts > 520)][0])})"
            )
        jumps = np.abs(np.diff(counts.astype(int)))
        big = np.flatnonzero(jumps > 25)
        if len(big):
            issues.append(
                f"{len(big)} saltos de más de 25 miembros entre snapshots consecutivos "
                f"(primero en {pd.Timestamp(idx.dates[big[0] + 1]).date()})"
            )
        gaps = np.diff(idx.dates).astype("timedelta64[D]").astype(int)
        long_gaps = np.flatnonzero(gaps > 90)
        if len(long_gaps):
            issues.append(
                f"{len(long_gaps)} huecos de más de 90 días entre snapshots "
                f"(el mayor: {int(gaps.max())} días)"
            )
        return issues

    # --------------------------------------------------------------- refresh

    def refresh(
        self,
        sources: Sequence[str] | Sequence[UniverseSource] | None = None,
        *,
        dry_run: bool = False,
        **kwargs: object,
    ) -> RefreshReport:
        """Actualiza el histórico desde fuentes vivas, **solo añadiendo** días nuevos.

        `sources` admite nombres (`"github"`, `"wikipedia"`, `"slickcharts"`) o
        instancias ya construidas. Sin argumento usa `sources.default_sources()`.

        Jamás reescribe una fecha ya registrada: las discrepancias en el tramo
        solapado se devuelven en `RefreshReport.divergences` para revisión humana. Si
        ninguna fuente responde lanza `ProviderUnavailable`.
        """
        resolved: list[UniverseSource] | None
        if sources is None:
            resolved = None
        else:
            resolved = [build_source(s) if isinstance(s, str) else s for s in sources]
        report = refresh_history_file(
            self._history_path, resolved, dry_run=dry_run, **kwargs  # type: ignore[arg-type]
        )
        self._identifiers = None
        return report

    # ------------------------------------------------------------ utilidades

    def restrict(self, tickers: Iterable[Ticker]) -> set[Ticker]:
        """Normaliza un conjunto de símbolos y descarta los que nunca estuvieron.

        Devolver el conjunto filtrado (y no un error) es lo correcto aquí: quien
        consulta suele venir de un proveedor de precios con su propio universo, y lo
        que necesita es la intersección explícita.
        """
        known = set(self.index.tickers)
        return {clean_symbol(t) for t in tickers} & known

    def __repr__(self) -> str:
        try:
            idx = self.index
        except ProviderUnavailable:  # pragma: no cover - repr defensivo
            return f"SP500Universe(history={self._history_path}, sin cargar)"
        return (
            f"SP500Universe({idx.n_snapshots} snapshots, "
            f"{idx.first_date.date()}→{idx.last_date.date()}, "
            f"{len(idx.tickers)} símbolos históricos)"
        )
