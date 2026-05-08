"""
Configuración de base de datos SQLite con SQLAlchemy.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from atp_predictor.core.paths import get_project_root

# Base de datos en la raíz del proyecto
DB_PATH = get_project_root() / "atp_tennis.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency para FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
