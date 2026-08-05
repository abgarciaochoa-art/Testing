"""Almacén append-only del recolector (`collector.storage`).

Parquet particionado por fecha, con tres propiedades que no son negociables en
un recolector cuyo valor es precisamente la disciplina point-in-time:

1. **Append-only.** Ningún fichero ya escrito se modifica jamás. Añadir datos
   escribe *part files* nuevos; no existe operación de sobreescritura. Es la
   misma disciplina que `SP500Universe.refresh()` (contrato §3.1): la historia
   registrada no se toca.
2. **Idempotencia por deduplicación.** Re-ejecutar la misma pasada el mismo día
   (un cron que reintenta, una máquina que se reinicia) no duplica filas: cada
   dataset declara su clave natural y `append` descarta las filas cuya clave ya
   está en la partición. La consecuencia deliberada es que **la primera captura
   del día gana**: una segunda foto horas más tarde no sustituye a la primera,
   porque serían dos vintages distintos disfrazados de uno.
3. **Integridad verificable.** Cada partición lleva un manifiesto con el número
   de filas y el SHA-256 de cada part file. `read(validate=True)` comprueba
   ambos y también que no haya ficheros parquet *fuera* del manifiesto (el
   residuo típico de un proceso interrumpido). Un panel que se acumula durante
   años en la máquina del usuario, sin nadie mirándolo, necesita detectar la
   corrupción silenciosa de disco en el momento de leer, no tres papers después.

`available_at` es obligatorio en todas las filas de todos los datasets: es la
columna que hace utilizable el panel con `pit.asof_join` y la razón de ser del
recolector (contrato §0.1).

Diseño de particiones
---------------------
Cada dataset declara su columna de partición (`DatasetSpec.partition_column`):
la fecha *del dato*, no la de la captura, salvo cuando coinciden. Ejemplos:

- `option_chain` particiona por `chain_date` (= sesión de la foto);
- `open_interest` por `oi_date` (la sesión T-1 cuyo cierre describe el OI
  capturado la mañana de T);
- `short_interest` por `settlement_date`: así la captura diaria "por si ha
  salido la publicación" es idempotente entre días — la publicación nueva crea
  partición nueva, y las repeticiones caen en la partición existente y se
  deduplican.

El formato en disco es Hive-style (`dataset/date=YYYY-MM-DD/part-*.parquet`),
legible por pyarrow/duckdb/spark sin este módulo delante.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from earnings_alpha.collector.snapshot import (
    CALENDAR_SNAPSHOT_COLUMNS,
    CONSENSUS_SNAPSHOT_COLUMNS,
    OFF_EXCHANGE_COLUMNS,
    OPEN_INTEREST_COLUMNS,
    OPTION_CHAIN_COLUMNS,
    SHORT_INTEREST_COLUMNS,
)
from earnings_alpha.data.cache import SystemWallClock, WallClock
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
)
from earnings_alpha.pit import TradingCalendar

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "MANIFEST_NAME",
    "STORAGE_FORMAT_VERSION",
    "DatasetSpec",
    "DATASET_SPECS",
    "AppendResult",
    "IntegrityProblem",
    "IntegrityReport",
    "SnapshotStore",
]

logger = logging.getLogger(__name__)

MANIFEST_NAME = "_MANIFEST.json"
STORAGE_FORMAT_VERSION = 1
"""Versión del formato del manifiesto. Al subirla, las particiones antiguas se
rechazan de forma ruidosa en vez de leerse con un esquema equivocado."""

_PART_RE = re.compile(r"^part-\d{4}-[0-9a-f]{8}\.parquet$")
_DATE_DIR_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Contrato de un dataset del recolector.

    `dedup_keys` es la clave natural: dos filas con la misma clave son *el mismo
    hecho* y la segunda se descarta. `partition_column` debe ser una columna de
    fecha (o timestamp normalizable a fecha) presente en todas las filas.
    """

    name: str
    dedup_keys: tuple[str, ...]
    partition_column: str
    required_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            msg = "el nombre del dataset no puede estar vacío"
            raise ConfigError(msg)
        if "available_at" not in self.required_columns:
            msg = f"el dataset {self.name!r} no declara `available_at`; sin él no es point-in-time"
            raise ConfigError(msg)
        missing_keys = [k for k in self.dedup_keys if k not in self.required_columns]
        if missing_keys:
            msg = f"claves de deduplicación fuera del esquema de {self.name!r}: {missing_keys}"
            raise ConfigError(msg)
        if self.partition_column not in self.required_columns:
            msg = (
                f"la columna de partición {self.partition_column!r} no está en el esquema "
                f"de {self.name!r}"
            )
            raise ConfigError(msg)


DATASET_SPECS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec(
            name="option_chain",
            dedup_keys=("chain_date", "ticker", "expiry", "right", "strike", "source"),
            partition_column="chain_date",
            required_columns=OPTION_CHAIN_COLUMNS,
        ),
        DatasetSpec(
            name="open_interest",
            dedup_keys=("oi_date", "ticker", "expiry", "right", "strike", "source"),
            partition_column="oi_date",
            required_columns=OPEN_INTEREST_COLUMNS,
        ),
        DatasetSpec(
            name="consensus",
            dedup_keys=("as_of", "ticker", "period_end", "source"),
            partition_column="as_of",
            required_columns=CONSENSUS_SNAPSHOT_COLUMNS,
        ),
        DatasetSpec(
            name="earnings_calendar",
            dedup_keys=("capture_date", "ticker", "period_end", "announced_at", "source"),
            partition_column="capture_date",
            required_columns=CALENDAR_SNAPSHOT_COLUMNS,
        ),
        DatasetSpec(
            name="off_exchange",
            dedup_keys=("trade_date", "ticker", "market", "source"),
            partition_column="trade_date",
            required_columns=OFF_EXCHANGE_COLUMNS,
        ),
        DatasetSpec(
            name="short_interest",
            dedup_keys=("settlement_date", "ticker", "source"),
            partition_column="settlement_date",
            required_columns=SHORT_INTEREST_COLUMNS,
        ),
    )
}
"""Datasets del recolector. La repetición de la misma clave de calendario en
particiones (capturas) sucesivas es deliberada: cada día es un vintage."""


@dataclass(frozen=True, slots=True)
class AppendResult:
    """Resultado de un `append` sobre una partición concreta."""

    dataset: str
    partition_date: dt.date
    rows_in: int
    rows_appended: int
    rows_duplicated: int
    path: Path | None
    """Fichero part escrito, o `None` si todo eran duplicados (no se escribe nada)."""


@dataclass(frozen=True, slots=True)
class IntegrityProblem:
    """Una anomalía detectada al verificar el almacén."""

    dataset: str
    partition: str
    file: str
    kind: str
    detail: str

    def describe(self) -> str:
        return f"[{self.dataset}/{self.partition}] {self.file}: {self.kind} — {self.detail}"


@dataclass(slots=True)
class IntegrityReport:
    """Informe de `SnapshotStore.verify()`."""

    partitions_checked: int = 0
    files_checked: int = 0
    rows_total: int = 0
    problems: list[IntegrityProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def describe(self) -> str:
        head = (
            f"particiones={self.partitions_checked} ficheros={self.files_checked} "
            f"filas={self.rows_total} problemas={len(self.problems)}"
        )
        if self.ok:
            return f"integridad OK ({head})"
        lines = [f"integridad CON PROBLEMAS ({head}):"]
        lines.extend(f"  - {p.describe()}" for p in self.problems)
        return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _as_partition_date(value: object) -> dt.date:
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if pd.isna(ts):
        msg = f"valor de partición no interpretable como fecha: {value!r}"
        raise DataQualityError(msg)
    return ts.date()


class SnapshotStore:
    """Almacén parquet append-only, particionado por fecha, con manifiesto.

    Parámetros
    ----------
    root:
        Directorio raíz (p. ej. ``data/collector``). Se crea si no existe.
    specs:
        Mapa de datasets admitidos; por defecto `DATASET_SPECS`. Un dataset no
        declarado se rechaza con `ConfigError`: el esquema es un contrato, no
        una sugerencia.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        specs: Mapping[str, DatasetSpec] | None = None,
        clock: WallClock | None = None,
    ) -> None:
        self.root = Path(root)
        self.specs: dict[str, DatasetSpec] = dict(specs or DATASET_SPECS)
        self.clock: WallClock = clock or SystemWallClock()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- rutas ----------------------------------------------------------------

    def spec_of(self, dataset: str) -> DatasetSpec:
        spec = self.specs.get(dataset)
        if spec is None:
            msg = f"dataset desconocido: {dataset!r}. Declarados: {sorted(self.specs)}"
            raise ConfigError(msg)
        return spec

    def dataset_dir(self, dataset: str) -> Path:
        return self.root / self.spec_of(dataset).name

    def partition_dir(self, dataset: str, day: dt.date) -> Path:
        return self.dataset_dir(dataset) / f"date={day.isoformat()}"

    # -- manifiesto -----------------------------------------------------------

    def _manifest_path(self, dataset: str, day: dt.date) -> Path:
        return self.partition_dir(dataset, day) / MANIFEST_NAME

    def _read_manifest(self, dataset: str, day: dt.date) -> dict[str, object]:
        path = self._manifest_path(dataset, day)
        if not path.exists():
            return {
                "version": STORAGE_FORMAT_VERSION,
                "dataset": dataset,
                "partition": day.isoformat(),
                "parts": [],
            }
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            msg = f"manifiesto ilegible en {path}: {exc}"
            raise DataQualityError(msg) from exc
        if int(raw.get("version", -1)) != STORAGE_FORMAT_VERSION:
            msg = (
                f"manifiesto de {dataset}/{day.isoformat()} con versión "
                f"{raw.get('version')!r} (actual {STORAGE_FORMAT_VERSION}); no se lee con un "
                "esquema que no le corresponde"
            )
            raise DataQualityError(msg)
        return raw

    def _write_manifest(self, dataset: str, day: dt.date, manifest: dict[str, object]) -> None:
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        _atomic_write_bytes(self._manifest_path(dataset, day), payload.encode("utf-8"))

    # -- validación de entrada ------------------------------------------------

    def _validate_frame(self, spec: DatasetSpec, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or len(frame) == 0:
            msg = (
                f"no se archiva un DataFrame vacío en {spec.name!r}: si la fuente no tiene "
                "datos debe lanzar (contrato §0.3), y si los tiene, esto es un bug"
            )
            raise DataQualityError(msg)
        missing = [c for c in spec.required_columns if c not in frame.columns]
        if missing:
            msg = f"faltan columnas del esquema de {spec.name!r}: {missing}"
            raise DataQualityError(msg)
        out = frame[list(spec.required_columns)].copy()
        for col in ("available_at", "captured_at"):
            series = pd.to_datetime(out[col], errors="coerce", utc=True)
            if series.isna().any():
                n = int(series.isna().sum())
                msg = (
                    f"{spec.name!r}: {n} filas con {col!r} nulo o no interpretable; "
                    "un dato sin fecha de disponibilidad no es point-in-time y no se archiva"
                )
                raise DataQualityError(msg)
            out[col] = series
        return out

    # -- escritura ------------------------------------------------------------

    def append(self, dataset: str, frame: pd.DataFrame) -> list[AppendResult]:
        """Añade filas nuevas; devuelve un resultado por partición afectada.

        El DataFrame puede abarcar varias fechas de partición (p. ej. una
        publicación de short interest con dos settlements): se parte y cada
        grupo se anexa a su partición. Las filas cuya clave ya existe se
        descartan; si en un grupo no queda ninguna fila nueva, **no se escribe
        ningún fichero** (idempotencia real, no un fichero vacío por pasada).
        """
        spec = self.spec_of(dataset)
        clean = self._validate_frame(spec, frame)
        partition_values = clean[spec.partition_column].map(_as_partition_date)
        results: list[AppendResult] = []
        for day, group in clean.groupby(partition_values.rename("__partition__"), sort=True):
            results.append(self._append_partition(spec, day, group.drop(columns=[], errors="ignore")))
        return results

    def _append_partition(
        self, spec: DatasetSpec, day: dt.date, group: pd.DataFrame
    ) -> AppendResult:
        rows_in = len(group)
        # Deduplicación intra-lote: la primera aparición gana, igual que entre lotes.
        group = group.drop_duplicates(subset=list(spec.dedup_keys), keep="first")
        existing_keys = self._existing_keys(spec, day)
        if existing_keys is not None and len(existing_keys):
            incoming = group[list(spec.dedup_keys)].astype(str).agg("|".join, axis=1)
            fresh_mask = ~incoming.isin(existing_keys)
            group = group[fresh_mask]
        rows_appended = len(group)
        rows_duplicated = rows_in - rows_appended
        if rows_appended == 0:
            logger.info(
                "%s/date=%s: 0 filas nuevas (%d duplicadas); no se escribe nada",
                spec.name,
                day.isoformat(),
                rows_duplicated,
            )
            return AppendResult(spec.name, day, rows_in, 0, rows_duplicated, None)

        directory = self.partition_dir(spec.name, day)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = self._read_manifest(spec.name, day)
        parts = list(manifest.get("parts", []))  # type: ignore[arg-type]
        seq = len(parts)
        filename = f"part-{seq:04d}-{uuid.uuid4().hex[:8]}.parquet"
        path = directory / filename

        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        try:
            group.reset_index(drop=True).to_parquet(tmp, engine="pyarrow", index=False)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

        collector_versions = sorted({str(v) for v in group["collector_version"].unique()})
        parts.append(
            {
                "file": filename,
                "rows": rows_appended,
                "sha256": _sha256_file(path),
                "written_at": self.clock.now().astimezone(dt.UTC).isoformat(),
                "collector_versions": collector_versions,
            }
        )
        manifest["parts"] = parts
        self._write_manifest(spec.name, day, manifest)
        return AppendResult(spec.name, day, rows_in, rows_appended, rows_duplicated, path)

    def _existing_keys(self, spec: DatasetSpec, day: dt.date) -> pd.Series | None:
        directory = self.partition_dir(spec.name, day)
        if not directory.exists():
            return None
        manifest = self._read_manifest(spec.name, day)
        keys: list[pd.Series] = []
        for part in manifest.get("parts", []):  # type: ignore[union-attr]
            path = directory / str(part["file"])
            if not path.exists():
                msg = (
                    f"{spec.name}/date={day.isoformat()}: el manifiesto declara "
                    f"{part['file']!r} y el fichero no existe; partición corrupta"
                )
                raise DataQualityError(msg)
            existing = pd.read_parquet(path, engine="pyarrow", columns=list(spec.dedup_keys))
            keys.append(existing.astype(str).agg("|".join, axis=1))
        if not keys:
            return pd.Series(dtype=object)
        return pd.concat(keys, ignore_index=True)

    # -- lectura --------------------------------------------------------------

    def partitions(self, dataset: str) -> list[dt.date]:
        """Fechas de partición existentes, ordenadas."""
        directory = self.dataset_dir(dataset)
        if not directory.exists():
            return []
        out: list[dt.date] = []
        for child in directory.iterdir():
            match = _DATE_DIR_RE.match(child.name)
            if match and child.is_dir():
                out.append(dt.date.fromisoformat(match.group(1)))
        return sorted(out)

    def read(
        self,
        dataset: str,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
        columns: Sequence[str] | None = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Lee las particiones en ``[start, end]`` verificando la integridad.

        Con `validate=True` (por defecto) cada part file debe coincidir con su
        manifiesto en SHA-256 y número de filas, no puede faltar ninguno y no
        puede haber parquet huérfanos. Sin filas en el rango ->
        `InsufficientHistory` (contrato §0.3).
        """
        spec = self.spec_of(dataset)
        days = [
            d
            for d in self.partitions(dataset)
            if (start is None or d >= start) and (end is None or d <= end)
        ]
        frames: list[pd.DataFrame] = []
        for day in days:
            frames.extend(self._read_partition(spec, day, columns=columns, validate=validate))
        if not frames:
            window = f"[{start}..{end}]" if (start or end) else "(todo el rango)"
            msg = (
                f"sin datos de {dataset!r} en {window}: el recolector aún no ha capturado "
                "ese rango o las particiones no existen"
            )
            raise InsufficientHistory(msg)
        out = pd.concat(frames, ignore_index=True)
        if validate and columns is None:
            dupes = out.duplicated(subset=list(spec.dedup_keys))
            if bool(dupes.any()):
                msg = (
                    f"{dataset!r}: {int(dupes.sum())} filas duplicadas por clave "
                    f"{spec.dedup_keys} tras leer; el almacén está corrupto o fue "
                    "manipulado fuera de este módulo"
                )
                raise DataQualityError(msg)
        sort_cols = [spec.partition_column, *spec.dedup_keys]
        sort_cols = list(dict.fromkeys(c for c in sort_cols if c in out.columns))
        return out.sort_values(sort_cols).reset_index(drop=True)

    def _read_partition(
        self,
        spec: DatasetSpec,
        day: dt.date,
        *,
        columns: Sequence[str] | None,
        validate: bool,
    ) -> list[pd.DataFrame]:
        directory = self.partition_dir(spec.name, day)
        manifest = self._read_manifest(spec.name, day)
        declared = {str(p["file"]): p for p in manifest.get("parts", [])}  # type: ignore[union-attr]
        on_disk = {p.name for p in directory.glob("*.parquet")}

        if validate:
            orphans = sorted(on_disk - set(declared))
            if orphans:
                msg = (
                    f"{spec.name}/date={day.isoformat()}: parquet fuera del manifiesto: "
                    f"{orphans}. Residuo probable de un proceso interrumpido; revisa con "
                    "`verify` y elimínalo a mano antes de leer"
                )
                raise DataQualityError(msg)
            missing = sorted(set(declared) - on_disk)
            if missing:
                msg = (
                    f"{spec.name}/date={day.isoformat()}: faltan ficheros declarados en el "
                    f"manifiesto: {missing}"
                )
                raise DataQualityError(msg)

        frames: list[pd.DataFrame] = []
        for name, meta in declared.items():
            path = directory / name
            if validate:
                digest = _sha256_file(path)
                if digest != str(meta["sha256"]):
                    msg = (
                        f"{spec.name}/date={day.isoformat()}: SHA-256 de {name} no coincide "
                        "con el manifiesto; el fichero está corrupto o fue modificado"
                    )
                    raise DataQualityError(msg)
            frame = pd.read_parquet(
                path, engine="pyarrow", columns=list(columns) if columns else None
            )
            if validate and columns is None and len(frame) != int(meta["rows"]):
                msg = (
                    f"{spec.name}/date={day.isoformat()}: {name} tiene {len(frame)} filas y el "
                    f"manifiesto declara {meta['rows']}"
                )
                raise DataQualityError(msg)
            frames.append(frame)
        return frames

    # -- diagnóstico ----------------------------------------------------------

    def verify(self, dataset: str | None = None) -> IntegrityReport:
        """Verifica checksums, recuentos y huérfanos de todo el almacén.

        A diferencia de `read`, no se detiene en el primer problema: recorre
        todo y devuelve la lista completa, que es lo útil para reparar.
        """
        report = IntegrityReport()
        names = [dataset] if dataset else sorted(self.specs)
        for name in names:
            if not self.dataset_dir(name).exists():
                continue
            for day in self.partitions(name):
                report.partitions_checked += 1
                directory = self.partition_dir(name, day)
                try:
                    manifest = self._read_manifest(name, day)
                except DataQualityError as exc:
                    report.problems.append(
                        IntegrityProblem(name, day.isoformat(), MANIFEST_NAME, "manifiesto", str(exc))
                    )
                    continue
                declared = {str(p["file"]): p for p in manifest.get("parts", [])}  # type: ignore[union-attr]
                on_disk = {p.name for p in directory.glob("*.parquet")}
                for orphan in sorted(on_disk - set(declared)):
                    report.problems.append(
                        IntegrityProblem(
                            name,
                            day.isoformat(),
                            orphan,
                            "huérfano",
                            "parquet presente en disco y ausente del manifiesto",
                        )
                    )
                for missing in sorted(set(declared) - on_disk):
                    report.problems.append(
                        IntegrityProblem(
                            name,
                            day.isoformat(),
                            missing,
                            "ausente",
                            "declarado en el manifiesto y no existe en disco",
                        )
                    )
                for fname, meta in declared.items():
                    path = directory / fname
                    if not path.exists():
                        continue
                    report.files_checked += 1
                    digest = _sha256_file(path)
                    if digest != str(meta["sha256"]):
                        report.problems.append(
                            IntegrityProblem(
                                name, day.isoformat(), fname, "sha256", "checksum no coincide"
                            )
                        )
                        continue
                    report.rows_total += int(meta["rows"])
        return report

    def missing_sessions(
        self,
        dataset: str,
        calendar: TradingCalendar,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> list[dt.date]:
        """Sesiones bursátiles del rango sin partición: los huecos del panel.

        Pensado para los datasets **diarios por sesión** (`option_chain`,
        `open_interest`, `off_exchange`); en los quincenales (`short_interest`)
        una sesión sin partición no es un hueco, es el calendario de FINRA.
        Sin rango explícito se usa `[primera partición, última partición]`.
        """
        have = set(self.partitions(dataset))
        if not have:
            return []
        lo = start or min(have)
        hi = end or max(have)
        sessions = calendar.sessions(lo, hi)
        return [ts.date() for ts in sessions if ts.date() not in have]

    def status(self) -> str:
        """Resumen legible del contenido del almacén (para el CLI)."""
        lines = [f"almacén: {self.root}"]
        for name in sorted(self.specs):
            days = self.partitions(name)
            if not days:
                lines.append(f"  {name}: (vacío)")
                continue
            rows = 0
            for day in days:
                manifest = self._read_manifest(name, day)
                rows += sum(int(p["rows"]) for p in manifest.get("parts", []))  # type: ignore[union-attr]
            lines.append(
                f"  {name}: {len(days)} particiones, {rows} filas, "
                f"{days[0].isoformat()} → {days[-1].isoformat()}"
            )
        return "\n".join(lines)
