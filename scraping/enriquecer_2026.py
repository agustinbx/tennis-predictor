import sys
from pathlib import Path
from datetime import date

import pandas as pd
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from surface_utils import detectar_superficie

# --- CONFIGURACIÓN ---
ARCHIVO_NUEVO = "atp_matches_2026_indetectable.csv" # Tu CSV flaco (recién bajado)
ARCHIVO_PERFILES = "perfiles_jugadores.pkl"         # Tu diccionario de datos (bajado antes)
ARCHIVO_SALIDA = "atp_matches_2026_full.csv"        # El CSV gordo final

# Los perfiles se generan a partir del histórico, cuyo último año completo
# es este. Los jugadores envejecen ~1 año por cada año que pasa desde ahí,
# en vez de sumar un valor fijo que queda desactualizado con el tiempo.
ANIO_REFERENCIA_PERFILES = 2024
AJUSTE_EDAD = date.today().year - ANIO_REFERENCIA_PERFILES

print("[INJECTION] INICIANDO ENRIQUECIMIENTO DE DATOS...")

# 1. CARGAR DATOS
try:
    df = pd.read_csv(ARCHIVO_NUEVO)
    perfiles = joblib.load(ARCHIVO_PERFILES)
    print(f"[DATA] Cargados {len(df)} partidos nuevos.")
    print(f"[DATA] Cargados {len(perfiles)} perfiles de jugadores.")
except Exception as e:
    print(f"[FAIL] Error cargando archivos: {e}")
    print("Asegúrate de tener 'atp_matches_2026_indetectable.csv' y 'perfiles_jugadores.pkl'")
    exit()

# 2. LISTAS PARA LOS DATOS NUEVOS
w_ht, w_age, w_rank, w_points, w_hand, w_ioc = [], [], [], [], [], []
l_ht, l_age, l_rank, l_points, l_hand, l_ioc = [], [], [], [], [], []
surfaces = []

print("[REFRESH] Cruzando datos...")

for index, row in df.iterrows():
    w_name = row['winner_name']
    l_name = row['loser_name']
    torneo = row['tourney_name']

    # --- A. SUPERFICIE ---
    # Única fuente de verdad para detectar superficie (ver surface_utils.py)
    surfaces.append(detectar_superficie(torneo))

    # --- B. DATOS DEL GANADOR ---
    if w_name in perfiles:
        p = perfiles[w_name]
        w_rank.append(p.get('rank', 100)) # Si no tiene rank, ponemos 100
        w_ht.append(p.get('ht', 185))     # Altura promedio 185
        w_age.append(p.get('age', 25) + AJUSTE_EDAD)
        w_points.append(p.get('points', 0)) # Puntos ATP actuales del perfil
        w_ioc.append(p.get('ioc', 'UNK'))
        w_hand.append('R') # Asumimos diestro si falta (dato menor)
    else:
        # JUGADOR NUEVO (ROOKIE): sin historial, 0 puntos es un valor real, no un placeholder
        w_rank.append(150)
        w_ht.append(185)
        w_age.append(22)
        w_points.append(0)
        w_ioc.append('UNK')
        w_hand.append('R')

    # --- C. DATOS DEL PERDEDOR ---
    if l_name in perfiles:
        p = perfiles[l_name]
        l_rank.append(p.get('rank', 100))
        l_ht.append(p.get('ht', 185))
        l_age.append(p.get('age', 25) + AJUSTE_EDAD)
        l_points.append(p.get('points', 0))
        l_ioc.append(p.get('ioc', 'UNK'))
        l_hand.append('R')
    else:
        l_rank.append(150)
        l_ht.append(185)
        l_age.append(22)
        l_points.append(0)
        l_ioc.append('UNK')
        l_hand.append('R')

# 3. AGREGAR COLUMNAS AL DATAFRAME
df['surface'] = surfaces
df['winner_ht'] = w_ht
df['winner_age'] = w_age
df['winner_rank'] = w_rank
df['winner_rank_points'] = w_points
df['winner_hand'] = w_hand
df['winner_ioc'] = w_ioc

df['loser_ht'] = l_ht
df['loser_age'] = l_age
df['loser_rank'] = l_rank
df['loser_rank_points'] = l_points
df['loser_hand'] = l_hand
df['loser_ioc'] = l_ioc

# Columnas que de verdad no se pueden reconstruir con los datos disponibles:
# NaN en vez de 0, para que no se confundan con un valor real (ver train.py,
# que usa pd.notna() para decidir si hay dato o no).
for col in ['match_num', 'best_of']:
    df[col] = np.nan

# 5. GUARDAR
df.to_csv(ARCHIVO_SALIDA, index=False)
print("\n" + "="*50)
print(f"[OK] ¡ENRIQUECIMIENTO COMPLETADO!")
print(f"[DOC] Archivo guardado: {ARCHIVO_SALIDA}")
print(f"[STATS] Ahora tienes Ranking, Altura y Edad estimados para 2026.")
print("="*50)