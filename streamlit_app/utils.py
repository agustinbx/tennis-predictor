"""
Utilidades compartidas para la app Streamlit.
"""
import sys
from pathlib import Path

# Agregar src al path para imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st
import pandas as pd
import requests
from typing import Optional, Dict, Any, List

from atp_predictor.core.paths import get_project_root, get_scraping_dir, get_models_dir

from atp_predictor.config import get_settings


# CSS para ocultar elementos de Streamlit
HIDE_ST_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


def apply_style():
    """Aplica estilos personalizados a la app."""
    st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)


def get_api_url() -> str:
    """Obtiene la URL de la API desde la configuración."""
    settings = get_settings()
    return settings.api_url


def load_perfiles() -> Optional[Dict[str, Any]]:
    """
    Carga los perfiles de jugadores desde el archivo PKL.    
    Returns:
        Diccionario con los perfiles o None si no se encuentra
    """
    import joblib
    
    possible_paths = [
        get_scraping_dir() / "perfiles_jugadores.pkl",
        get_models_dir() / "perfiles_jugadores.pkl",
        get_project_root() / "scraping" / "perfiles_jugadores.pkl",
    ]
    
    for path in possible_paths:
        if path.exists():
            return joblib.load(path)
    
    return None


def load_ranking() -> Optional[pd.DataFrame]:
    """
    Carga el ranking ATP desde CSV.
    
    Returns:
        DataFrame con el ranking o None si no se encuentra
    """
    possible_paths = [
        get_scraping_dir() / "ranking_2026.csv",
        get_project_root() / "scraping" / "ranking_2026.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)
    
    return None


def load_comparison_results() -> Optional[pd.DataFrame]:
    """
    Carga los resultados de comparación de modelos.
    
    Returns:
        DataFrame con los resultados o None si no se encuentra
    """
    possible_paths = [
        get_project_root() / "resultados_comparacion.csv",
        get_models_dir() / "resultados_comparacion.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)
    
    return None


def load_feature_importance() -> Optional[pd.DataFrame]:
    """
    Carga la importancia de variables.    
    Returns:
        DataFrame con la importancia o None si no se encuentra
    """
    possible_paths = [
        get_project_root() / "importancia_real.csv",
        get_models_dir() / "importancia_real.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)
    
    return None


@st.cache_data(ttl=300)
def get_players_from_api() -> List[str]:
    """
    Obtiene la lista de jugadores desde la API.
    
    Returns:
        Lista de nombres de jugadores
    """
    try:
        response = requests.get(f"{get_api_url()}/players/", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    
    # Fallback: cargar desde perfiles locales
    perfiles = load_perfiles()
    if perfiles:
        return sorted(perfiles.keys())
    
    return []


@st.cache_data(ttl=300)
def get_player_profile(player_name: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene el perfil de un jugador desde la API.
    
    Args:
        player_name: Nombre del jugador
    
    Returns:
        Diccionario con el perfil o None si no se encuentra
    """
    try:
        response = requests.get(
            f"{get_api_url()}/players/{player_name}",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    
    # Fallback: cargar desde perfiles locales
    perfiles = load_perfiles()
    if perfiles and player_name in perfiles:
        profile = perfiles[player_name]
        # Normalizar keys del perfil local para que coincidan con la UI
        # El perfil usa 'rank', 'points', 'age', 'ioc'
        # La UI espera 'ranking', 'puntos', 'edad', 'nacionalidad'
        normalized = dict(profile)  # copiar todo
        if 'rank' in normalized and 'ranking' not in normalized:
            normalized['ranking'] = normalized['rank']
        if 'points' in normalized and 'puntos' not in normalized:
            normalized['puntos'] = normalized['points']
        if 'age' in normalized and 'edad' not in normalized:
            normalized['edad'] = normalized['age']
        if 'ioc' in normalized and 'nacionalidad' not in normalized:
            normalized['nacionalidad'] = normalized['ioc']
        return normalized
    
    return None


@st.cache_data(ttl=300)
def get_player_surface_stats(player_name: str, surface: str) -> float:
    """
    Obtiene el win rate de un jugador en una superficie.
    
    Args:
        player_name: Nombre del jugador
        surface: Superficie ('Hard', 'Clay', 'Grass')
    
    Returns:
        Win rate (0-1)
    """
    try:
        response = requests.get(
            f"{get_api_url()}/stats/{player_name}/{surface}",
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("win_rate", 0.5)
    except requests.exceptions.RequestException:
        pass
    
    return 0.5


def predict_match(
    jugador_1: str,
    jugador_2: str,
    superficie: str,
    pais_torneo: str = "NEUTRAL",
    modelo: str = "XGBoost",
    fatiga_1: int = 0,
    fatiga_2: int = 0,
    descanso_1: int = 14,
    descanso_2: int = 14
) -> Optional[Dict[str, Any]]:
    """
    Realiza una predicción de partido.

    Args:
        jugador_1: Nombre del jugador 1
        jugador_2: Nombre del jugador 2
        superficie: Superficie ('Hard', 'Clay', 'Grass')
        pais_torneo: País del torneo
        modelo: Modelo a usar ('XGBoost' o 'Logistic Regression')
        fatiga_1: Fatiga acumulada del jugador 1
        fatiga_2: Fatiga acumulada del jugador 2
        descanso_1: Días de descanso desde el último partido del jugador 1
        descanso_2: Días de descanso desde el último partido del jugador 2

    Returns:
        Diccionario con la predicción o None si hay error
    """
    payload = {
        "jugador_1": jugador_1,
        "jugador_2": jugador_2,
        "superficie": superficie,
        "pais_torneo": pais_torneo,
        "modelo": modelo,
        "fatiga_1": fatiga_1,
        "fatiga_2": fatiga_2,
        "descanso_1": descanso_1,
        "descanso_2": descanso_2
    }
    
    try:
        response = requests.post(
            f"{get_api_url()}/predict",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    
    return None


def extract_player_name_from_url(url: str) -> str:
    """
    Extrae el nombre del jugador desde una URL de ATP.    
    Args:
        url: URL del perfil de ATP    
    Returns:
        Nombre del jugador
    """
    try:
        slug = str(url).split('/')[5]
        return slug.replace('-', ' ').title()
    except:
        return ""
