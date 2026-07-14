# project_io.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple

# Directorio base = carpeta donde vive ESTE archivo (o sea, tu proyecto)
BASE_DIR = Path(__file__).resolve().parent

# Carpetas estándar
DATA_DIR   = (BASE_DIR / "data").resolve()
GRAPHS_DIR = (BASE_DIR / "graphs").resolve()

def ensure_dirs() -> Tuple[Path, Path]:
    """Crea data/ y graphs/ si no existen. Devuelve (DATA_DIR, GRAPHS_DIR)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, GRAPHS_DIR

def data_path(*parts: str) -> Path:
    """Ruta dentro de data/ (crea carpetas si faltan)."""
    ensure_dirs()
    p = DATA_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def graphs_path(*parts: str) -> Path:
    """Ruta dentro de graphs/ (crea carpetas si faltan)."""
    ensure_dirs()
    p = GRAPHS_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
