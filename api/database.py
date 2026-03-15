from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# URL de conexión a SQLite
# El archivo 'atp_tennis.db' se creará en la carpeta raíz del proyecto
SQLALCHEMY_DATABASE_URL = "sqlite:///./atp_tennis.db"

# Para PostgreSQL sería algo así:
# SQLALCHEMY_DATABASE_URL = "postgresql://usuario:contraseña@localhost/nombre_bd"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    # check_same_thread=False es necesario solo para SQLite en FastAPI
    connect_args={"check_same_thread": False} 
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
