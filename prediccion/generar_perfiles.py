import pandas as pd
import joblib
import numpy as np

print("👤 GENERANDO PERFILES (V5.0 - SOLUCIÓN TOTAL)...")

try:
    # 1. Cargar CSV
    df = pd.read_csv("historial_tenis_COMPLETO.csv")
    
    # --- A. LIMPIEZA Y FORMATO ---
    df['tourney_id'] = df['tourney_id'].astype(str)
    
    # Asegurar que columnas numéricas no tengan texto basura
    cols_check = ['winner_age', 'winner_ht', 'loser_age', 'loser_ht', 'winner_rank', 'loser_rank']
    for col in cols_check:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- B. REPARACIÓN DE FECHAS (2026) ---
    def reparar_fecha(row):
        fecha = pd.to_numeric(row['tourney_date'], errors='coerce')
        if fecha > 19900000: return fecha
        # Intentar sacar año del ID
        try:
            year = row['tourney_id'].split('-')[0]
            if len(year) == 4 and year.isdigit():
                return int(year) * 10000 + 101 # 20260101
        except: pass
        return 0

    df['date_fix'] = df.apply(reparar_fecha, axis=1)

    # --- C. MAPEO DE RONDAS (CORREGIDO SEGÚN TU CSV) ---
    # Usamos .strip() más abajo para quitar espacios extra (ej: "Final ")
    round_map = {
        # Clasificación (Valen menos)
        'Q1': 1,
        'Q2': 2,
        'Q3': 3,
        
        # Cuadro Principal (Valen más)
        'Round of 128': 10,
        'Round of 64': 20,
        'Round of 32': 30,
        'Round of 16': 40,
        'Quarterfinals': 50,
        'Semifinals': 60,
        'Final': 70, # La más importante
        'The Final': 70, # Por si acaso
        
        # Mantenemos las abreviaturas viejas por compatibilidad con años anteriores
        'R128': 10, 'R64': 20, 'R32': 30, 'R16': 40, 'QF': 50, 'SF': 60, 'F': 70, 'W': 80
    }

    # Limpiamos espacios en blanco del nombre de la ronda (ej: "Final " -> "Final")
    df['round_clean'] = df['round'].astype(str).str.strip()
    
    # Mapeamos
    df['round_val'] = df['round_clean'].map(round_map).fillna(0)

    # --- D. ORDENAMIENTO MAESTRO ---
    # 1. Fecha (Lo viejo primero)
    # 2. Torneo
    # 3. Ronda (Final al último)
    df = df.sort_values(by=['date_fix', 'tourney_id', 'round_val'])
    
    print(f"   Procesando {len(df)} partidos ordenados...")

    # --- E. PROCESAMIENTO CON MEMORIA ---
    perfiles = {}
    racha_tracker = {}
    
    # "Cache" para recordar datos si vienen vacíos
    bio_cache = {} 

    for index, row in df.iterrows():
        w = row['winner_name']
        l = row['loser_name']
        score = row['score']
        torneo = row['tourney_name']
        ronda = row['round']
        fecha = row['tourney_date']
        
        # --- 1. GESTIÓN DEL HISTORIAL (RACHA) ---
        rw = racha_tracker.get(w, [])
        rl = racha_tracker.get(l, [])
        
        # A. Datos para el GANADOR
        match_w = {
            'resultado': 'W',      # Won
            'rival': l,            # El rival fue el perdedor
            'score': score,        # El score (siempre está visto desde el ganador)
            'torneo': torneo,
            'ronda': ronda
        }
        
        # B. Datos para el PERDEDOR
        match_l = {
            'resultado': 'L',      # Lost
            'rival': w,            # El rival fue el ganador
            'score': score,        # El score
            'torneo': torneo,
            'ronda': ronda
        }
        
        rw.append(match_w)
        rl.append(match_l)
        
        # Mantenemos solo los últimos 5
        if len(rw) > 5: rw.pop(0)
        if len(rl) > 5: rl.pop(0)
        
        racha_tracker[w] = rw
        racha_tracker[l] = rl
        
        # --- 2. DATOS BIO (Igual que antes) ---
        # (Resumido para no ocupar espacio, la lógica es la misma de la V4.0)
        mem_w = bio_cache.get(w, {'age': 25, 'ht': 185, 'ioc': 'UNK', 'points': 0, 'rank': 500})
        if pd.notna(row.get('winner_age')) and row['winner_age'] > 10: mem_w['age'] = row['winner_age']
        if pd.notna(row.get('winner_ht')) and row['winner_ht'] > 100: mem_w['ht'] = row['winner_ht']
        if pd.notna(row.get('winner_ioc')) and str(row['winner_ioc']) != '0': mem_w['ioc'] = row['winner_ioc']
        if pd.notna(row.get('winner_rank')): mem_w['rank'] = row['winner_rank']
        if pd.notna(row.get('winner_rank_points')): mem_w['points'] = row['winner_rank_points']
        bio_cache[w] = mem_w; perfiles[w] = mem_w.copy()

        mem_l = bio_cache.get(l, {'age': 25, 'ht': 185, 'ioc': 'UNK', 'points': 0, 'rank': 500})
        if pd.notna(row.get('loser_age')) and row['loser_age'] > 10: mem_l['age'] = row['loser_age']
        if pd.notna(row.get('loser_ht')) and row['loser_ht'] > 100: mem_l['ht'] = row['loser_ht']
        if pd.notna(row.get('loser_ioc')) and str(row['loser_ioc']) != '0': mem_l['ioc'] = row['loser_ioc']
        if pd.notna(row.get('loser_rank')): mem_l['rank'] = row['loser_rank']
        if pd.notna(row.get('loser_rank_points')): mem_l['points'] = row['loser_rank_points']
        bio_cache[l] = mem_l; perfiles[l] = mem_l.copy()

    # --- F. GUARDADO FINAL ---
    for jugador, datos in perfiles.items():
        historial = racha_tracker.get(jugador, [])
        
        # Calculamos momentum numérico (Win = 1, Loss = 0)
        victorias = sum(1 for x in historial if x['resultado'] == 'W')
        momentum = victorias / len(historial) if historial else 0.5
        
        perfiles[jugador]['momentum'] = momentum
        perfiles[jugador]['last_5'] = historial

    # Debug
    print("\n🔎 VERIFICACIÓN FINAL:")
    if 'Novak Djokovic' in perfiles:
        racha = perfiles['Novak Djokovic']['last_5']
        iconos = ["✅" if x==1 else "🔴" for x in racha]
        print(f"   Novak Djokovic (Últimos 5): {iconos}")
        
    if 'Carlos Alcaraz' in perfiles:
        p = perfiles['Carlos Alcaraz']
        print(f"   Carlos Alcaraz -> Edad: {p['age']}, País: {p['ioc']}")

    joblib.dump(perfiles, 'perfiles_jugadores.pkl')
    print("\n✅ Archivo actualizado con éxito.")

except Exception as e:
    print(f"❌ Error: {e}")