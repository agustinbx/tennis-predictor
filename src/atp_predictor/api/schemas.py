"""
Esquemas Pydantic para validación de API.
"""
from pydantic import BaseModel
from typing import List, Optional


class PlayerBase(BaseModel):
    nombre: str
    puntos: Optional[int] = 0
    ranking: Optional[int] = 9999
    edad: Optional[float] = 0.0
    altura: Optional[int] = 0
    momentum: Optional[float] = 0.0
    nacionalidad: Optional[str] = None
    
    class Config:
        from_attributes = True


class MatchPredictionRequest(BaseModel):
    """Request para predicción de partido."""
    jugador_1: str
    jugador_2: str
    superficie: str  # Hard, Clay, Grass
    pais_torneo: str = "NEUTRAL"
    modelo: str = "XGBoost"
    fatiga_1: int = 0
    fatiga_2: int = 0
    descanso_1: int = 14
    descanso_2: int = 14


class MatchPredictionResponse(BaseModel):
    """Response de predicción de partido."""
    ganador: str
    confianza: float
    probabilidad_j1: float
    probabilidad_j2: float
    modelo_utilizado: str
    explicacion: List[str]
    h2h: str
