import sys
import os

# Agregamos la ruta principal para que Python encuentre los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from api.database import SessionLocal, engine, Base
from api.models_db import PlayerProfile, MatchStats

# 1. Crear las tablas en la BD
print("⚙️ Creando tablas en la base de datos...")
Base.metadata.create_all(bind=engine)

def load_data():
    db = SessionLocal()
    
    # 2. Cargar perfiles
    try:
        print("📥 Cargando perfiles_jugadores.pkl...")
        perfiles = joblib.load("scraping/perfiles_jugadores.pkl")
        
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
            
        print(f"✅ Se insertaron {len(perfiles)} jugadores.")

    except FileNotFoundError:
        print("❌ Archivo scraping/perfiles_jugadores.pkl no encontrado.")
        
    # 3. Cargar estadísticas de superficie
    try:
        print("📥 Cargando stats_superficie_v2.pkl...")
        stats = joblib.load("prediccion/stats_superficie_v2.pkl")
        
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
            
        print(f"✅ Se insertaron {count} estadísticas de superficie.")
    except FileNotFoundError:
        print("❌ Archivo prediccion/stats_superficie_v2.pkl no encontrado.")

    # Guardar cambios
    db.commit()
    db.close()
    print("🚀 Migración finalizada con éxito!")

if __name__ == "__main__":
    load_data()
