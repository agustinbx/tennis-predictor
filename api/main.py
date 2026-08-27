"""
API REST para predicción de partidos de tenis ATP.
"""
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import joblib
import pandas as pd
import logging

# Imports del paquete
from atp_predictor.core.paths import get_models_dir
from atp_predictor.api.database import get_db, engine, Base
from atp_predictor.api.models import PlayerProfile, MatchStats
from atp_predictor.api.schemas import MatchPredictionRequest
from atp_predictor.config import get_settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear tablas si no existen
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ATP Predictor API",
    description="API para predecir partidos de tenis ATP usando Machine Learning",
    version="1.0.0"
)

# --- CARGA DE MODELOS ---
def load_models():
    """Carga los modelos ML y trackers desde disco."""
    models_dir = get_models_dir()
    
    models = {}
    required_files = {
        'xgboost': 'modelo_xgboost_final.pkl',
        'logistic': 'modelo_logistico_final.pkl',
        'scaler': 'scaler_final.pkl',
        'h2h_tracker': 'h2h_tracker.pkl',
        'h2h_weighted_tracker': 'h2h_weighted_tracker.pkl',
        'elo_tracker': 'elo_tracker.pkl',
        'elo_surface_tracker': 'elo_surface_tracker.pkl',
        'clutch_tracker': 'clutch_tracker.pkl',
        'surface_stats': 'stats_superficie_v2.pkl',
        'surface_momentum_tracker': 'surface_momentum_tracker.pkl',
    }
    
    for key, filename in required_files.items():
        path = models_dir / filename
        if path.exists():
            models[key] = joblib.load(path)
            logger.info(f"[OK] Cargado: {filename}")
        else:
            models[key] = {} if key != 'scaler' else None
            logger.warning(f"[WARN] No encontrado: {filename}")
    
    return models


# Cargar modelos al inicio
try:
    MODELS = load_models()
    logger.info("[OK] Modelos ML y trackers cargados exitosamente.")
except Exception as e:
    logger.error(f"[FAIL] Error cargando modelos: {e}")
    MODELS = {}


# --- FUNCIÓN DE EXPLICABILIDAD ---
def generar_explicacion(diffs: dict, ganador: str, nombre1: str, nombre2: str) -> List[str]:
    """Genera explicación narrativa de por qué el modelo predice un ganador."""
    
    es_j1 = (ganador == nombre1)
    perdedor = nombre2 if es_j1 else nombre1
    
    razones = []
    v_rank = v_elo = v_skill = v_clutch = v_momentum = v_h2h = v_fatigue = ""
    
    if es_j1:
        if diffs['diff_rank'] >= 50: v_rank = f"le saca muchos puestos de ventaja en el ranking ATP"
        elif diffs['diff_rank'] <= -50: v_rank = f"la IA confía en él pese a estar mucho peor posicionado en el ranking ATP"
        
        if diffs['diff_elo'] >= 80: v_elo = f"llega con un nivel tenístico reciente muy superior (ELO)"
        
        if diffs['diff_skill'] >= 0.15: v_skill = f"casi no pierde en esta superficie"
        elif diffs['diff_skill'] >= 0.08: v_skill = f"es estadísticamente superior en este tipo de pista"
        
        if diffs['diff_clutch'] >= 0.08: v_clutch = f"tiene nervios de acero bajo presión comparado a su rival"
        
        if diffs['diff_momentum'] >= 0.30: v_momentum = f"viene impulsado por una excelente racha de victorias"
        
        if diffs['diff_h2h'] >= 2: v_h2h = f"el historial directo juega a su favor"
        
        if diffs['diff_fatigue'] <= -2: v_fatigue = f"llegó al partido con menos desgaste físico en este torneo"
        elif diffs['diff_fatigue'] >= 3: v_fatigue = f"la IA lo da como ganador a pesar de llegar con más desgaste de sets encima"
    else:
        # Misma lógica pero invertida para jugador 2
        if diffs['diff_rank'] <= -50: v_rank = f"le saca muchos puestos de ventaja en el ranking ATP"
        elif diffs['diff_rank'] >= 50: v_rank = f"la IA confía en él pese a estar mucho peor posicionado en el ranking ATP"
        
        if diffs['diff_elo'] <= -80: v_elo = f"llega con un nivel tenístico reciente muy superior (ELO)"
        
        if diffs['diff_skill'] <= -0.15: v_skill = f"casi no pierde en esta superficie"
        elif diffs['diff_skill'] <= -0.08: v_skill = f"es estadísticamente superior en este tipo de pista"
        
        if diffs['diff_clutch'] <= -0.08: v_clutch = f"tiene nervios de acero bajo presión comparado a su rival"
        
        if diffs['diff_momentum'] <= -0.30: v_momentum = f"viene impulsado por una excelente racha de victorias"
        
        if diffs['diff_h2h'] <= -2: v_h2h = f"el historial directo juega a su favor"
        
        if diffs['diff_fatigue'] >= 2: v_fatigue = f"llegó al partido con menos desgaste físico en este torneo"
        elif diffs['diff_fatigue'] <= -3: v_fatigue = f"la IA lo da como ganador a pesar de llegar con más desgaste de sets encima"

    argumentos = [a for a in [v_rank, v_elo, v_skill, v_clutch, v_momentum, v_h2h, v_fatigue] if a != ""]
    
    if len(argumentos) == 0:
        parrafo = f"El predictor pronostica un duelo extremadamente parecido entre {ganador} y {perdedor}, inclinándose por {ganador} basado en micro-diferencias estadísticas invisibles a simple vista."
    elif len(argumentos) == 1:
        parrafo = f"El modelo se inclina por {ganador} principalmente porque {argumentos[0]} en comparación a {perdedor}."
    else:
        ultimo = argumentos.pop()
        lista_format = ", ".join(argumentos) + f", y {ultimo}"
        parrafo = f"El modelo se inclina por {ganador} principalmente porque {lista_format}."
    
    return [parrafo]


# --- ENDPOINTS ---

@app.get("/")
def read_root():
    """Health check."""
    return {"message": "ATP Predictor API funcionando [START]", "version": "1.0.0"}


@app.get("/players/", response_model=List[str])
def get_all_players(db: Session = Depends(get_db)):
    """Obtiene la lista de todos los jugadores."""
    jugadores = db.query(PlayerProfile.nombre).order_by(PlayerProfile.nombre).all()
    return [j[0] for j in jugadores]


@app.get("/players/{nombre}")
def get_player(nombre: str, db: Session = Depends(get_db)):
    """Obtiene el perfil de un jugador."""
    jugador = db.query(PlayerProfile).filter(PlayerProfile.nombre == nombre).first()
    if jugador is None:
        raise HTTPException(status_code=404, detail="Jugador no encontrado en la base de datos")
    return jugador


@app.get("/stats/{nombre}/{superficie}")
def get_player_surface_stats(nombre: str, superficie: str, db: Session = Depends(get_db)):
    """Obtiene el win rate de un jugador en una superficie."""
    stat = db.query(MatchStats).filter(
        MatchStats.nombre == nombre,
        MatchStats.superficie == superficie
    ).first()
    
    if stat:
        return {"win_rate": stat.win_rate}
    return {"win_rate": 0.5}


@app.post("/predict")
def predict_match(req: MatchPredictionRequest, db: Session = Depends(get_db)):
    """Predice el ganador de un partido."""
    
    # Verificar que los modelos están cargados
    if not MODELS.get('xgboost') or not MODELS.get('scaler'):
        raise HTTPException(status_code=503, detail="Modelos no disponibles. Verifique los archivos .pkl")
    
    # Obtener datos de la BD
    p1 = get_player(req.jugador_1, db)
    p2 = get_player(req.jugador_2, db)
    
    # Obtener skills en la superficie
    skill1 = get_player_surface_stats(req.jugador_1, req.superficie, db)["win_rate"]
    skill2 = get_player_surface_stats(req.jugador_2, req.superficie, db)["win_rate"]
    
    # Obtener valores de trackers
    elo1 = MODELS['elo_tracker'].get(req.jugador_1, 1500)
    elo2 = MODELS['elo_tracker'].get(req.jugador_2, 1500)

    # ELO específico de la superficie del partido (ver core/features.py::SurfaceEloTracker)
    elo_surface_por_superficie = MODELS['elo_surface_tracker'].get(req.superficie, {})
    elo_surf1 = elo_surface_por_superficie.get(req.jugador_1, 1500)
    elo_surf2 = elo_surface_por_superficie.get(req.jugador_2, 1500)

    c1 = MODELS['clutch_tracker'].get(req.jugador_1, 0.5)
    c2 = MODELS['clutch_tracker'].get(req.jugador_2, 0.5)
    
    # Datos de perfil
    a1 = p1.edad or 25
    a2 = p2.edad or 25
    m1 = p1.momentum or 0.5
    m2 = p2.momentum or 0.5
    r1 = p1.ranking or 9999
    r2 = p2.ranking or 9999
    pts1 = p1.puntos or 0
    pts2 = p2.puntos or 0
    ht1 = p1.altura or 185
    ht2 = p2.altura or 185
    
    # Cálculo de localía
    home_1 = 1 if p1.nacionalidad == req.pais_torneo else 0
    home_2 = 1 if p2.nacionalidad == req.pais_torneo else 0
    diff_home = home_1 - home_2
    
    # Cálculo de H2H
    p1_sort, p2_sort = sorted([req.jugador_1, req.jugador_2])
    key = (p1_sort, p2_sort)
    record = MODELS['h2h_tracker'].get(key, [0, 0])
    
    if req.jugador_1 == p1_sort:
        h2h_1_wins = record[0]
        h2h_2_wins = record[1]
    else:
        h2h_1_wins = record[1]
        h2h_2_wins = record[0]
    
    diff_h2h = h2h_1_wins - h2h_2_wins

    # H2H ponderado por recencia (ver core/features.py::WeightedH2HTracker)
    record_weighted = MODELS['h2h_weighted_tracker'].get(key, [0.0, 0.0])

    if req.jugador_1 == p1_sort:
        h2h_1_reciente = record_weighted[0]
        h2h_2_reciente = record_weighted[1]
    else:
        h2h_1_reciente = record_weighted[1]
        h2h_2_reciente = record_weighted[0]

    diff_h2h_reciente = h2h_1_reciente - h2h_2_reciente

    # Momentum reciente específico de esta superficie (ver
    # core/features.py::SurfaceMomentumTracker)
    momentum_surf1 = MODELS['surface_momentum_tracker'].get((req.jugador_1, req.superficie), 0.5)
    momentum_surf2 = MODELS['surface_momentum_tracker'].get((req.jugador_2, req.superficie), 0.5)

    # Preparar datos para predicción
    input_data = pd.DataFrame([{
        'diff_elo': elo1 - elo2,
        'diff_rank': r2 - r1,
        'diff_points': pts1 - pts2,
        'diff_clutch': c1 - c2,
        'diff_age': a1 - a2,
        'diff_ht': ht1 - ht2,
        'diff_skill': skill1 - skill2,
        'diff_fatigue': req.fatiga_1 - req.fatiga_2,
        'diff_momentum': m1 - m2,
        'diff_h2h': diff_h2h,
        'diff_home': diff_home,
        'diff_elo_surface': elo_surf1 - elo_surf2,
        'diff_descanso': req.descanso_1 - req.descanso_2,
        'diff_h2h_reciente': diff_h2h_reciente,
        'diff_momentum_superficie': momentum_surf1 - momentum_surf2
    }])
    
    try:
        input_scaled = MODELS['scaler'].transform(input_data)
        
        # Seleccionar modelo
        active_model = MODELS['xgboost'] if "XGBoost" in req.modelo else MODELS.get('logistic', MODELS['xgboost'])
        
        prob = active_model.predict_proba(input_scaled)[0]
        prob_j1 = prob[1]
        
        ganador = req.jugador_1 if prob_j1 > 0.5 else req.jugador_2
        confianza = float(prob_j1 if prob_j1 > 0.5 else 1 - prob_j1)
        
        dict_diffs = input_data.iloc[0].to_dict()
        explicacion = generar_explicacion(dict_diffs, ganador, req.jugador_1, req.jugador_2)
        
        return {
            "ganador": ganador,
            "confianza": confianza,
            "probabilidad_j1": float(prob_j1),
            "probabilidad_j2": float(1 - prob_j1),
            "modelo_utilizado": req.modelo,
            "explicacion": explicacion,
            "h2h": f"{req.jugador_1} {h2h_1_wins} - {h2h_2_wins} {req.jugador_2}"
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción interna: {str(e)}")


def run_server():
    """Función para ejecutar el servidor desde línea de comandos."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    run_server()
