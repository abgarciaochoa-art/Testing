"""Empaqueta los dossiers de caídas y de Noise Band en un zip para otro chat.

Estructura del zip (``traspaso_estrategias/``): ``LEEME.md`` y ``PROMPT_OTRO_CHAT.md`` de esta
carpeta, ``1_caidas/`` (``research/caidas/dossier/``: DOSSIER.md, datos/, graficas/),
``2_noise_band/`` (``research/noise_band/dossier/``, con ``referencia.py``) y ``codigo/``: una
copia del código para leerlo o ejecutarlo (``pyproject.toml``, el paquete ``earnings_alpha`` y
los ``.py`` y README de los dos estudios; los datos se descargan de fuentes públicas con sus
``datos.py``). El prompt solo de caídas que hay en su carpeta no se incluye: lo sustituye el
de aquí, que cubre las dos.

Uso: python research/traspaso/empaquetar.py [--salida RUTA.zip]
(por defecto, data/cache/traspaso/traspaso_estrategias.zip)
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

AQUI = Path(__file__).parent
ROOT = AQUI.parents[1]
PARTES = {"1_caidas": ROOT / "research" / "caidas" / "dossier",
          "2_noise_band": ROOT / "research" / "noise_band" / "dossier"}


def codigo() -> list[Path]:
    """Lo necesario para leer o ejecutar los dos estudios (sin datos ni resultados)."""
    fuentes = [ROOT / "pyproject.toml", *sorted((ROOT / "earnings_alpha").rglob("*.py"))]
    for estudio in ("caidas", "noise_band", "traspaso"):
        carpeta = ROOT / "research" / estudio
        fuentes += sorted([*carpeta.glob("*.py"), *carpeta.glob("*.md")])
    return [f for f in fuentes if "__pycache__" not in f.parts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=ROOT / "data" / "cache" / "traspaso" / "traspaso_estrategias.zip")
    salida = ap.parse_args().salida
    salida.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("LEEME.md", "PROMPT_OTRO_CHAT.md"):
            z.write(AQUI / f, f"traspaso_estrategias/{f}")
            n += 1
        for destino, origen in PARTES.items():
            for f in sorted(origen.rglob("*")):
                if f.is_file() and f.name != "PROMPT_OTRO_CHAT.md" and "__pycache__" not in f.parts:
                    z.write(f, f"traspaso_estrategias/{destino}/{f.relative_to(origen)}")
                    n += 1
        for f in codigo():
            z.write(f, f"traspaso_estrategias/codigo/{f.relative_to(ROOT)}")
            n += 1
    print(f"{n} archivos → {salida} ({salida.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
