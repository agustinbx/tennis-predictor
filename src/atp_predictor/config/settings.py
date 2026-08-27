"""
Configuración centralizada del proyecto.
Lee variables de entorno con valores por defecto.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno."""
    
    # Proyecto
    project_name: str = Field(default="ATP Predictor", alias="PROJECT_NAME")
    debug: bool = Field(default=False, alias="DEBUG")
    
    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8002, alias="API_PORT")
    api_url: str = Field(default="http://localhost:8002", alias="API_URL")
    
    # Base de datos
    db_path: str = Field(default="atp_tennis.db", alias="DB_PATH")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Nota: las rutas de directorios y archivos de modelos NO viven acá.
    # `core/paths.py` es la única fuente de verdad para eso (get_models_dir,
    # get_scraping_dir, etc.) — evita tener dos definiciones que puedan
    # divergir (ver core/paths.py::get_models_dir, que apunta a models/).

    # Scraping
    scraping_delay: float = Field(default=2.0, alias="SCRAPING_DELAY")
    scraping_headless: bool = Field(default=False, alias="SCRAPING_HEADLESS")
    
    # Streamlit
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Cachea la configuración para evitar lecturas repetidas."""
    return Settings()
