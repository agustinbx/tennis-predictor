"""
Utilidades para manejo de paths del proyecto.
"""
from pathlib import Path
from functools import lru_cache


@lru_cache()
def get_project_root() -> Path:
    """
    Obtiene el directorio raíz del proyecto.
    
    Busca desde la ubicación de este archivo hacia arriba
    hasta encontrar un directorio con pyproject.toml.
    """
    current = Path(__file__).resolve()
    
    # Buscar hacia arriba hasta encontrar pyproject.toml
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Fallback: 4 niveles hacia arriba desde este archivo
    # Este archivo está en src/atp_predictor/core/paths.py
    return Path(__file__).parent.parent.parent.parent


def get_src_root() -> Path:
    """Obtiene el directorio src/."""
    return get_project_root() / "src"


def get_data_dir() -> Path:
    """Obtiene el directorio de datos."""
    return get_project_root() / "data"


def get_processed_data_dir() -> Path:
    """Obtiene el directorio de datos procesados (data/processed/)."""
    path = get_project_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_external_data_dir() -> Path:
    """Obtiene el directorio de datos externos de terceros (data/external/)."""
    path = get_project_root() / "data" / "external"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    """Obtiene el directorio de modelos (models/)."""
    return get_project_root() / "models"


def get_scraping_dir() -> Path:
    """Obtiene el directorio de scraping."""
    return get_project_root() / "scraping"


def get_api_dir() -> Path:
    """Obtiene el directorio de la API."""
    return get_project_root() / "api"


def ensure_dir(path: Path) -> Path:
    """Crea el directorio si no existe."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Alias para compatibilidad
PROJECT_ROOT = get_project_root()
