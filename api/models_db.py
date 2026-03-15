from sqlalchemy import Column, Integer, String, Float, Boolean, JSON
from .database import Base

class PlayerProfile(Base):
    __tablename__ = "jugadores"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, unique=True, index=True)
    slug = Column(String)  # ej: carlos-alcaraz
    puntos = Column("points", Integer, nullable=True)
    ranking = Column("rank", Integer, nullable=True)
    edad = Column("age", Float, nullable=True)
    altura = Column("ht", Integer, nullable=True)
    momentum = Column("momentum", Float, nullable=True)
    nacionalidad = Column("ioc", String, nullable=True)
    # Algunos campos son JSON para guardar listas o diccionarios complejos que vengan del PKL
    last_5 = Column(JSON, nullable=True)
    
    # Estadísticas base
    aces = Column(Integer, nullable=True)
    df = Column(Integer, nullable=True) # Dobles faltas
    serve_win = Column(Float, nullable=True)
    bp_saved = Column(Float, nullable=True)
    service_hold = Column(Float, nullable=True)
    
class MatchStats(Base):
    __tablename__ = "estadisticas_superficie"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True)
    superficie = Column(String, index=True) # Hard, Clay, Grass
    win_rate = Column(Float)
