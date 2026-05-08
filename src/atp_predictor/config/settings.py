"""
Configuración centralizada del proyecto.
Lee variables de entorno con valores por defecto.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from functools import lru_cache
from pathlib import Path


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
    
    # Directorios
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    scraping_dir: Path = Field(default_factory=lambda: Path("scraping"))
    
    # Modelos ML
    xgboost_model: str = "modelo_xgboost_final.pkl"
    logistic_model: str = "modelo_logistico_final.pkl"
    scaler_model: str = "scaler_final.pkl"
    h2h_tracker: str = "h2h_tracker.pkl"
    elo_tracker: str = "elo_tracker.pkl"
    clutch_tracker: str = "clutch_tracker.pkl"
    surface_stats: str = "stats_superficie_v2.pkl"
    profiles: str = "perfiles_jugadores.pkl"
    
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
    
    def get_model_path(self, filename: str) -> Path:
        """Obtiene la ruta completa a un archivo de modelo."""
        return self.models_dir / filename
    
    def get_data_path(self, filename: str) -> Path:
        """Obtiene la ruta completa a un archivo de datos."""
        return self.data_dir / filename


@lru_cache()
def get_settings() -> Settings:
    """Cachea la configuración para evitar lecturas repetidas."""
    return Settings()
