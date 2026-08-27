import sys
import os

# Agregamos la ruta principal para que Python encuentre los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from api.database import SessionLocal, engine, Base
from atp_predictor.api.models import PlayerProfile, MatchStats
from atp_predictor.core.paths import get_models_dir, get_scraping_dir

# 1. Crear las tablas en la BD
print("[CONFIG] Creando tablas en la base de datos...")
Base.metadata.create_all(bind=engine)

def load_data():
    db = SessionLocal()
    
    # 2. Cargar perfiles
    try:
        print("[UPLOAD] Cargando perfiles_jugadores.pkl...")
        perfiles = joblib.load(get_scraping_dir() / "perfiles_jugadores.pkl")
        
        # Limpiamos tabla antes de insertar (para evitar duplicados al correr varias veces)
        db.query(PlayerProfile).delete()
        
        for nombre, datos in perfiles.items():
            jugador = PlayerProfile(
                nombre=nombre,
                slug=datos.get('slug', ''),
                puntos=datos.get('points', 0),
                ranking=datos.get('rank', 9999),
                edad=datos.get('age', 0.0),
                altura=datos.get('ht', 0),
                momentum=datos.get('momentum', 0.0),
                nacionalidad=datos.get('ioc', ''),
                last_5=datos.get('last_5', []),
                aces=datos.get('aces', 0),
                df=datos.get('df', 0),
                serve_win=datos.get('serve_win', 0.0),
                bp_saved=datos.get('bp_saved', 0.0),
                service_hold=datos.get('service_hold', 0.0)
            )
            db.add(jugador)
            
        print(f"[OK] Se insertaron {len(perfiles)} jugadores.")

    except FileNotFoundError:
        print("[FAIL] Archivo scraping/perfiles_jugadores.pkl no encontrado.")

    # 3. Cargar estadísticas de superficie
    try:
        print("[UPLOAD] Cargando stats_superficie_v2.pkl...")
        stats = joblib.load(get_models_dir() / "stats_superficie_v2.pkl")
        
        db.query(MatchStats).delete()
        
        count = 0
        for (nombre, superficie), win_rate in stats.items():
            stat = MatchStats(
                nombre=nombre,
                superficie=superficie,
                win_rate=win_rate
            )
            db.add(stat)
            count += 1
            
        print(f"[OK] Se insertaron {count} estadísticas de superficie.")
    except FileNotFoundError:
        print("[FAIL] Archivo models/stats_superficie_v2.pkl no encontrado.")

    # Guardar cambios
    db.commit()
    db.close()
    print("[START] Migración finalizada con éxito!")

if __name__ == "__main__":
    load_data()
