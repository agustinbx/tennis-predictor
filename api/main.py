from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import joblib
import pandas as pd
import os

from .database import get_db, engine, Base
from . import models_db

from pydantic import BaseModel

# Crear tablas si no existen
models_db.Base.metadata.create_all(bind=engine)

app = FastAPI(title="ATP Predictor API", description="API para predecir partidos de tenis", version="1.0.0")

# --- CARGA DE MODELOS ---
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_proyecto = os.path.dirname(ruta_script)

try:
    path_xgb = os.path.join(ruta_proyecto, 'prediccion', 'modelo_xgboost_final.pkl')
    path_log = os.path.join(ruta_proyecto, 'prediccion', 'modelo_logistico_final.pkl')
    path_scaler = os.path.join(ruta_proyecto, 'prediccion', 'scaler_final.pkl')
    
    model_xgb = joblib.load(path_xgb)
    model_log = joblib.load(path_log)
    scaler = joblib.load(path_scaler)
    print("✅ Modelos ML cargados en memoria exitosamente.")
except Exception as e:
    model_xgb = None
    model_log = None
    scaler = None
    print(f"❌ Error cargando modelos ML: {e}")

# --- SCHEMAS (Pydantic) para validación ---

class PlayerBase(BaseModel):
    nombre: str
    puntos: int | None = 0
    ranking: int | None = 9999
    edad: float | None = 0.0
    altura: int | None = 0
    momentum: float | None = 0.0
    nacionalidad: str | None = None
    
    class Config:
        from_attributes = True

class MatchPredictionRequest(BaseModel):
    jugador_1: str
    jugador_2: str
    superficie: str  # Hard, Clay, Grass
    pais_torneo: str
    modelo: str = "XGBoost" # XGBoost o Logistic
    fatiga_1: int = 0
    fatiga_2: int = 0
    h2h_1: int = 0
    h2h_2: int = 0
    
# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "ATP Predictor API funcionando 🚀"}

@app.get("/players/", response_model=List[str])
def get_all_players(db: Session = Depends(get_db)):
    # Devolveremos la lista de nombres para que Streamlit arme el selectbox
    jugadores = db.query(models_db.PlayerProfile.nombre).order_by(models_db.PlayerProfile.nombre).all()
    # jugadores es una lista de tuplas: [('Carlos Alcaraz',), ('Novak Djokovic',)]
    return [j[0] for j in jugadores]

@app.get("/players/{nombre}")
def get_player(nombre: str, db: Session = Depends(get_db)):
    jugador = db.query(models_db.PlayerProfile).filter(models_db.PlayerProfile.nombre == nombre).first()
    if jugador is None:
        raise HTTPException(status_code=404, detail="Jugador no encontrado en la base de datos")
    return jugador

@app.get("/stats/{nombre}/{superficie}")
def get_player_surface_stats(nombre: str, superficie: str, db: Session = Depends(get_db)):
    stat = db.query(models_db.MatchStats).filter(
        models_db.MatchStats.nombre == nombre,
        models_db.MatchStats.superficie == superficie
    ).first()
    
    if stat:
        return {"win_rate": stat.win_rate}
    return {"win_rate": 0.5} # Default si no hay datos

@app.post("/predict")
def predict_match(req: MatchPredictionRequest, db: Session = Depends(get_db)):
    # 1. Obtener datos de la BD
    p1 = get_player(req.jugador_1, db)
    p2 = get_player(req.jugador_2, db)
    
    # 2. Obtener skills en la superficie
    skill1 = get_player_surface_stats(req.jugador_1, req.superficie, db)["win_rate"]
    skill2 = get_player_surface_stats(req.jugador_2, req.superficie, db)["win_rate"]
    
    # 3. Cálculos
    home1 = 1 if p1.nacionalidad == req.pais_torneo else 0
    home2 = 1 if p2.nacionalidad == req.pais_torneo else 0
    
    # Manejar nulos por las dudas
    r1 = p1.ranking or 9999
    r2 = p2.ranking or 9999
    pts1 = p1.puntos or 0
    pts2 = p2.puntos or 0
    a1 = p1.edad or 25
    a2 = p2.edad or 25
    h1 = p1.altura or 180
    h2 = p2.altura or 180
    m1 = p1.momentum or 0.5
    m2 = p2.momentum or 0.5
    
    diff_h2h = req.h2h_1 - req.h2h_2
    
    # Validar que los campos de Streamlit crucen con el modelo
    # 'diff_rank', 'diff_rank_points', 'diff_age', 'diff_ht', 'diff_skill', 'diff_home', 'diff_fatigue', 'diff_momentum', 'diff_h2h'
    
    input_data = pd.DataFrame([{
        'diff_rank': r2 - r1,  # Si p1 es 1 y p2 es 10, es 9 positivo a favor de P1 (Misma lógica original)
        'diff_rank_points': pts1 - pts2,
        'diff_age': a1 - a2,
        'diff_ht': h1 - h2,
        'diff_skill': skill1 - skill2,
        'diff_home': home1 - home2,
        'diff_fatigue': req.fatiga_1 - req.fatiga_2,
        'diff_momentum': m1 - m2,
        'diff_h2h': diff_h2h
    }])
    
    try:
        input_scaled = scaler.transform(input_data)
        
        # Seleccionamos modelo
        active_model = model_xgb if "XGBoost" in req.modelo else model_log
        
        prob = active_model.predict_proba(input_scaled)[0]
        prob_j1 = prob[1] # Probabilidad de que gane 1 (True)
        
        ganador = req.jugador_1 if prob_j1 > 0.5 else req.jugador_2
        confianza = float(prob_j1 if prob_j1 > 0.5 else 1 - prob_j1)
        
        return {
            "ganador": ganador,
            "confianza": confianza,
            "probabilidad_j1": float(prob_j1),
            "probabilidad_j2": float(1 - prob_j1),
            "modelo_utilizado": req.modelo
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción interna: {str(e)}")
        
