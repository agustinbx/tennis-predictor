"""
Modelos ORM para la base de datos.
"""
from sqlalchemy import Column, Integer, String, Float, JSON
from sqlalchemy.orm import Session

from .database import Base


class PlayerProfile(Base):
    """Perfil de un jugador de tenis."""
    __tablename__ = "jugadores"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, unique=True, index=True)
    slug = Column(String)
    puntos = Column("points", Integer, nullable=True)
    ranking = Column("rank", Integer, nullable=True)
    edad = Column("age", Float, nullable=True)
    altura = Column("ht", Integer, nullable=True)
    momentum = Column(Float, nullable=True)
    nacionalidad = Column("ioc", String, nullable=True)
    last_5 = Column(JSON, nullable=True)
    
    # Estadísticas base
    aces = Column(Integer, nullable=True)
    df = Column(Integer, nullable=True)
    serve_win = Column(Float, nullable=True)
    bp_saved = Column(Float, nullable=True)
    service_hold = Column(Float, nullable=True)


class MatchStats(Base):
    """Estadísticas de partidos por superficie."""
    __tablename__ = "estadisticas_superficie"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True)
    superficie = Column(String, index=True)
    win_rate = Column(Float)
