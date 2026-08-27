import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from surface_utils import detectar_superficie

print("[GEAR] INICIANDO CORRECCIÓN DE RANKINGS...")

# --- ARCHIVOS DE ENTRADA ---
# Tu archivo con los partidos de 2025 y 2026 (puede ser el raw o el master)
ARCHIVO_PARTIDOS = "atp_matches_2026_full.csv" 
# Tu archivo con el ranking actual que scrapeamos antes
ARCHIVO_RANKING = "ranking_2026.csv"
# Archivo de salida limpio
ARCHIVO_SALIDA = "atp_matches_2026_corregido.csv"

if not Path(ARCHIVO_PARTIDOS).exists() or not Path(ARCHIVO_RANKING).exists():
    # Sin scrape 2026 disponible (p.ej. un runner de CI, que no corre
    # Selenium): no es un error, simplemente no hay nada nuevo que corregir.
    # fusionar_historico_final.py sigue funcionando solo con el histórico.
    print(f"[SKIP] No encuentro '{ARCHIVO_PARTIDOS}' y/o '{ARCHIVO_RANKING}' — nada que corregir, sigo sin fallar.")
    sys.exit(0)

try:
    # 1. CARGAR DATOS
    df = pd.read_csv(ARCHIVO_PARTIDOS)
    df_rank = pd.read_csv(ARCHIVO_RANKING)
    print(f"[OK] Partidos cargados: {len(df)}")
    print(f"[OK] Ranking cargado: {len(df_rank)}")

    # ---------------------------------------------------------
    # PASO 1: SUPERFICIE (solo si falta) [STADIUM]
    # ---------------------------------------------------------
    # La superficie ya la determina enriquecer_2026.py (única fuente de
    # verdad, ver surface_utils.py). Acá solo completamos si por algún
    # motivo faltara, en vez de recalcularla de nuevo con otra lógica.
    if 'surface' not in df.columns:
        df['surface'] = np.nan
    faltantes = df['surface'].isna()
    if faltantes.any():
        print(f"[GLOBE] Completando superficie faltante en {faltantes.sum()} partidos...")
        df.loc[faltantes, 'surface'] = df.loc[faltantes, 'tourney_name'].apply(detectar_superficie)

    conteo = df['surface'].value_counts()
    print(f"   [STATS] Superficies: Clay={conteo.get('Clay',0)}, Grass={conteo.get('Grass',0)}, Hard={conteo.get('Hard',0)}")

    # ---------------------------------------------------------
    # PASO 2: INYECTAR RANKING ACTUAL A 2026 [TOP]
    # ---------------------------------------------------------
    print("[INJECTION] Inyectando Ranking Actual a los partidos de 2026...")

    # Creamos un diccionario rápido: {'Carlos Alcaraz': 2, 'Jannik Sinner': 1}
    # Usamos 'player_slug' porque el scraper de ranking lo guardó así
    ranking_dict = df_rank.set_index('player')['rank'].to_dict()

    # Función para buscar ranking
    def get_current_rank(nombre, ranking_actual):
        # Si el jugador está en el top 500 actual, devolvemos su rank
        if nombre in ranking_dict:
            return ranking_dict[nombre]
        # Si tiene ranking viejo (del archivo), lo mantenemos. Si no, ponemos 500
        if pd.notna(ranking_actual) and ranking_actual > 0:
            return ranking_actual
        return 500 # Valor por defecto para desconocidos

    # Filtramos solo los partidos de 2026 (o todos si prefieres usar el rank actual para todo)
    # Aquí lo aplicamos a TODO el archivo 2025/2026 para que la IA sepa el nivel "actual" del jugador
    # Si prefieres solo 2026, cambia a: df[df['tourney_date'].astype(str).str.startswith('2026')]
    
    # Actualizamos Winner Rank
    df['winner_rank'] = df.apply(lambda row: get_current_rank(row['winner_name'], row.get('winner_rank', 0)), axis=1)
    
    # Actualizamos Loser Rank
    df['loser_rank'] = df.apply(lambda row: get_current_rank(row['loser_name'], row.get('loser_rank', 0)), axis=1)

    # ---------------------------------------------------------
    # PASO 3: LIMPIEZA FINAL Y GUARDADO [SAVE]
    # ---------------------------------------------------------
    
    # Rellenamos columnas faltantes con ceros para que no falle la fusión final
    cols_zero = ['winner_rank_points', 'loser_rank_points', 'match_num']
    for col in cols_zero:
        if col not in df.columns:
            df[col] = 0
            
    df.to_csv(ARCHIVO_SALIDA, index=False)
    
    print("\n" + "="*50)
    print("[DONE] ¡CORRECCIÓN COMPLETADA!")
    print(f"[DATA] Archivo listo: {ARCHIVO_SALIDA}")
    print("   -> Superficies corregidas.")
    print("   -> Ranking 2026 actualizado.")
    print("="*50)

except Exception as e:
    print(f"[FAIL] Error: {e}")
    sys.exit(1)