#!/usr/bin/env python
"""
Generador de perfiles de jugadores.

Este script procesa el historial de partidos y genera perfiles
con estadísticas actualizadas para cada jugador.
"""
import sys
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atp_predictor.core.paths import get_project_root, get_scraping_dir, get_models_dir, get_processed_data_dir


def find_csv_file(filename: str) -> Path:
    """Busca un archivo CSV en múltiples ubicaciones."""
    project_root = get_project_root()
    possible_paths = [
        get_processed_data_dir() / filename,
        project_root / filename,
        get_scraping_dir() / filename,
        get_models_dir() / filename,
        project_root / "scraping" / filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(f"No se encontró {filename}")


def load_historial() -> pd.DataFrame:
    """Carga el historial de partidos."""
    csv_path = find_csv_file("historialTenis.csv")
    print(f"[DATA] Cargando historial desde: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['tourney_id'] = df['tourney_id'].astype(str)
    
    # Limpiar columnas numéricas
    numeric_cols = ['winner_age', 'winner_ht', 'loser_age', 'loser_ht', 'winner_rank', 'loser_rank']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def process_profiles(df: pd.DataFrame) -> dict:
    """Procesa el historial y genera perfiles de jugadores."""
    print("[USER] Generando perfiles...")
    
    # Mapeo de rondas para ordenamiento
    round_map = {
        'Q1': 1, 'Q2': 2, 'Q3': 3,
        'R128': 10, 'R64': 20, 'R32': 30, 'R16': 40, 'QF': 50, 'SF': 60, 'F': 70, 'W': 80,
        'Round of 128': 10, 'Round of 64': 20, 'Round of 32': 30, 'Round of 16': 40,
        'Quarterfinals': 50, 'Semifinals': 60, 'Final': 70, 'The Final': 70,
        'Round Robin': 5,
    }
    
    # Limpiar rondas
    df['round_clean'] = df['round'].astype(str).str.strip()
    df['round_val'] = df['round_clean'].map(round_map).fillna(0)
    
    # Ordenar por fecha
    def create_sort_key(row):
        fecha = pd.to_numeric(row.get('tourney_date', 0), errors='coerce')
        if pd.notna(fecha) and fecha > 19900000:
            return fecha * 100
        
        try:
            tid = str(row.get('tourney_id', ''))
            parts = tid.split('-')
            year = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 2026
            idx = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0
            return (year * 1000000) + (idx * 100)
        except:
            return 2026999900
    
    df['sort_key'] = df.apply(create_sort_key, axis=1)
    df = df.sort_values(by=['sort_key', 'round_val'])
    
    # Procesar perfiles
    profiles = {}
    bio_cache = {}
    match_count = {}
    # Guardar detalle de cada partido para last_5
    match_history = {}  # player -> list of match dicts

    for idx, row in df.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        score = row.get('score', '')
        tourney = row.get('tourney_name', '')
        round_name = row.get('round', '')
        
        match_count[winner] = match_count.get(winner, 0) + 1
        match_count[loser] = match_count.get(loser, 0) + 1
        
        # Guardar detalle de partido para last_5
        for player, result, opponent in [(winner, 'W', loser), (loser, 'L', winner)]:
            if player not in match_history:
                match_history[player] = []
            match_history[player].append({
                'resultado': result,
                'rival': opponent,
                'score': str(score) if pd.notna(score) else '',
                'ronda': str(round_name) if pd.notna(round_name) else '',
                'torneo': str(tourney) if pd.notna(tourney) else '',
            })
            # Mantener solo los ultimos 5
            if len(match_history[player]) > 5:
                match_history[player].pop(0)
        
        # Actualizar bio cache
        for player, prefix in [(winner, 'winner'), (loser, 'loser')]:
            if player not in bio_cache:
                bio_cache[player] = {'age': 25, 'ht': 185, 'ioc': 'UNK', 'points': 0, 'rank': 500}
            
            age = row.get(f'{prefix}_age')
            ht = row.get(f'{prefix}_ht')
            ioc = row.get(f'{prefix}_ioc')
            rank = row.get(f'{prefix}_rank')
            points = row.get(f'{prefix}_rank_points')
            
            if pd.notna(age) and age > 10:
                bio_cache[player]['age'] = age
            if pd.notna(ht) and ht > 100:
                bio_cache[player]['ht'] = ht
            if pd.notna(ioc) and str(ioc) != '0':
                bio_cache[player]['ioc'] = ioc
            if pd.notna(rank):
                bio_cache[player]['rank'] = rank
            if pd.notna(points):
                bio_cache[player]['points'] = points
        
        # Guardar perfiles
        profiles[winner] = bio_cache[winner].copy()
        profiles[loser] = bio_cache[loser].copy()
    
    # Calcular momentum y last_5 con datos reales
    for player in profiles:
        history = match_history.get(player, [])
        results = [1 if m['resultado'] == 'W' else 0 for m in history]
        profiles[player]['momentum'] = sum(results) / len(results) if results else 0.5
        profiles[player]['last_5'] = history[-5:] if history else []
    
    return profiles, match_count


def inject_advanced_stats(profiles: dict, match_count: dict) -> dict:
    """Inyecta estadísticas avanzadas y ranking actualizado."""
    print("[INJECTION] Inyectando estadísticas avanzadas...")
    
    project_root = get_project_root()
    
    # Cargar estadísticas avanzadas
    try:
        stats_path = find_csv_file("estadisticas_jugadores_avanzadas.csv")
        df_stats = pd.read_csv(stats_path)
        stats_dict = df_stats.set_index('player').to_dict('index')
    except FileNotFoundError:
        print("[WARN] No se encontró estadisticas_jugadores_avanzadas.csv")
        stats_dict = {}
    
    # Cargar ranking actualizado
    try:
        ranking_path = find_csv_file("ranking_2026.csv")
        df_ranking = pd.read_csv(ranking_path)
        df_ranking = df_ranking.drop_duplicates(subset=['player'], keep='first')
        
        def extract_name(url):
            try:
                slug = str(url).split('/')[5]
                return slug.replace('-', ' ').title()
            except:
                return ""
        
        df_ranking['player_real'] = df_ranking['url_perfil'].apply(extract_name)
        ranking_dict = df_ranking.set_index('player_real').to_dict('index')
        print("[OK] Ranking actualizado cargado")
    except FileNotFoundError:
        print("[WARN] No se encontró ranking_2026.csv")
        ranking_dict = {}
    
    # Inyectar datos
    for player, data in profiles.items():
        # Actualizar ranking
        if player in ranking_dict:
            profiles[player]['rank'] = ranking_dict[player].get('rank', 500)
            profiles[player]['points'] = ranking_dict[player].get('points', 0)
        
        # Actualizar estadísticas avanzadas
        if player in stats_dict:
            n_matches = match_count.get(player, 1)
            extra = stats_dict[player]
            profiles[player]['serve_win'] = extra.get('serve_win_pct', 65.0)
            profiles[player]['bp_saved'] = extra.get('bp_saved_pct', 60.0)
            profiles[player]['service_hold'] = extra.get('service_hold_pct', 75.0)
            profiles[player]['aces'] = extra.get('aces_avg', 0.0) / n_matches
            profiles[player]['df'] = extra.get('df_avg', 0.0) / n_matches
        else:
            profiles[player]['serve_win'] = 65.0
            profiles[player]['bp_saved'] = 60.0
            profiles[player]['service_hold'] = 75.0
            profiles[player]['aces'] = 0.0
            profiles[player]['df'] = 0.0
    
    return profiles


def main():
    """Función principal."""
    print("="*60)
    print("[USER] GENERANDO PERFILES DE JUGADORES")
    print("="*60)
    
    # Cargar datos
    df = load_historial()
    print(f"[STATS] {len(df)} partidos cargados")
    
    # Procesar perfiles
    profiles, match_count = process_profiles(df)
    print(f"[USERS] {len(profiles)} jugadores procesados")
    
    # Inyectar datos adicionales
    profiles = inject_advanced_stats(profiles, match_count)
    
    # Verificación
    if 'Jannik Sinner' in profiles:
        p = profiles['Jannik Sinner']
        print(f"\n[SEARCH] Verificación - Jannik Sinner:")
        print(f"   Ranking: {p.get('rank')}")
        print(f"   Puntos: {p.get('points')}")
        print(f"   Momentum: {p.get('momentum', 0):.2%}")
    
    # Guardar
    output_path = get_scraping_dir() / "perfiles_jugadores.pkl"
    joblib.dump(profiles, output_path)
    
    print(f"\n[OK] Perfiles guardados en: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
