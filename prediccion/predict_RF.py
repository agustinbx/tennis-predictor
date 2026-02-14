import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("🏟️ Iniciando Entrenamiento con FACTOR LOCALÍA...")
# -------------------------------------------------------------------------
# 1. CARGAR PARTIDOS
# -------------------------------------------------------------------------
try:
    df = pd.read_csv("historial_tenis.csv")
    df['minutes'] = df['minutes'].fillna(100) # Rellenar nulos
    # ORDENAR CRONOLÓGICAMENTE ES VITAL PARA EL MOMENTUM
    df = df.sort_values(by=['tourney_date', 'tourney_id', 'match_num'])
except:
    print("❌ Falta el archivo CSV.")
    exit()

# -------------------------------------------------------------------------
# 2. DETECTAR PAÍS DE LOS TORNEOS (PARA CALCULAR LOCALÍA)
# -------------------------------------------------------------------------
def detectar_pais_por_nombre(nombre_torneo):
    # Convertimos a string y mayúsculas para evitar errores
    t = str(nombre_torneo).upper()
    
    # MAPA DE DETECCIÓN (Palabra Clave -> Código IOC)
    # Agregamos las sedes más importantes del circuito
    if any(x in t for x in ['MADRID', 'BARCELONA', 'VALENCIA', 'SEVILLE', 'MALLORCA']): return 'ESP'
    if any(x in t for x in ['PARIS', 'ROLAND GARROS', 'MONTPELLIER', 'MARSEILLE', 'LYON', 'METZ']): return 'FRA'
    if any(x in t for x in ['US OPEN', 'INDIAN WELLS', 'MIAMI', 'CINCINNATI', 'WASHINGTON', 'HOUSTON', 'DALLAS', 'DELRAY']): return 'USA'
    if any(x in t for x in ['WIMBLEDON', 'LONDON', 'QUEENS', 'EASTBOURNE', 'MANCHESTER']): return 'GBR'
    if any(x in t for x in ['AUSTRALIAN OPEN', 'MELBOURNE', 'BRISBANE', 'SYDNEY', 'ADELAIDE', 'PERTH']): return 'AUS'
    if any(x in t for x in ['ROME', 'ROMA', 'TURIN', 'MILAN', 'FLORENCE']): return 'ITA'
    if any(x in t for x in ['BUENOS AIRES', 'CORDOBA']): return 'ARG'
    if any(x in t for x in ['HAMBURG', 'HALLE', 'MUNICH', 'STUTTGART', 'BERLIN']): return 'GER'
    if any(x in t for x in ['RIO', 'SAO PAULO']): return 'BRA'
    if any(x in t for x in ['ACAPULCO', 'LOS CABOS']): return 'MEX'
    if any(x in t for x in ['TORONTO', 'MONTREAL', 'VANCOUVER']): return 'CAN'
    if any(x in t for x in ['SHANGHAI', 'BEIJING', 'CHENGDU', 'ZHUHAI']): return 'CHN'
    
    return 'NEUTRAL' # Si no estamos seguros, no damos ventaja a nadie

# Aplicamos la función a la columna 'tourney_name' que YA TENEMOS
df['tourney_ioc'] = df['tourney_name'].apply(detectar_pais_por_nombre)


# -------------------------------------------------------------------------
# 4. INGENIERÍA DE VARIABLES (SKILL + LOCALÍA)
# -------------------------------------------------------------------------
print("📊 Calculando variables...")

# A. Skill por superficie (Usamos tasa de victorias como proxy de habilidad en esa superficie)
wins = df.groupby(['winner_name', 'surface']).size().reset_index(name='wins')
wins.columns = ['player', 'surface', 'wins'] # Renombramos para facilitar el merge posterior
losses = df.groupby(['loser_name', 'surface']).size().reset_index(name='losses') # Unimos victorias y derrotas
losses.columns = ['player', 'surface', 'losses']
stats = pd.merge(wins, losses, on=['player', 'surface'], how='outer').fillna(0) # Calculamos tasa de victorias solo para jugadores con al menos 5 partidos en esa superficie
stats['total'] = stats['wins'] + stats['losses'] # Filtramos para quedarnos solo con jugadores con al menos 5 partidos en esa superficie
stats = stats[stats['total'] >= 5] # Calculamos tasa de victorias
stats['win_rate'] = stats['wins'] / stats['total'] # Creamos un diccionario para acceder rápidamente a la tasa de victorias por jugador y superficie. El resultado se guarda en 'stats_superficie_v2.pkl' para usarlo luego en la app. El modelo usará esta tasa de victorias como una medida de "habilidad" del jugador en esa superficie específica.
stats_dict = stats.set_index(['player', 'surface'])['win_rate'].to_dict() # Guardamos el diccionario para usarlo luego en la app
joblib.dump(stats_dict, 'stats_superficie_v2.pkl') # Guardamos el diccionario para usarlo luego en la app

def get_skill(player, surface):
    return stats_dict.get((player, surface), 0.5)

# ------------------------------------------------------------------------------
#  CÁLCULO DE FATIGA ACUMULADA POR TORNEO (PARA USARLO COMO FEATURE EN EL MODELO)
# -------------------------------------------------------------------------------
# Rellenamos minutos vacíos con el promedio (aprox 100 min) para no perder datos
df['minutes'] = df['minutes'].fillna(100)
# Ordenamos por fecha y número de partido para que el cálculo cronológico sea correcto
df = df.sort_values(by=['tourney_date', 'tourney_id', 'match_num'])

print("⏳ Calculando desgaste físico histórico (esto puede tardar unos segundos)...")

# Diccionario para guardar el cansancio: {(id_torneo, nombre_jugador): minutos_acumulados}
fatiga_tracker = {}
racha_tracker = {}  # {player: [1, 0, 1, 1, 0]}  (1=Ganó, 0=Perdió)

fatiga_ganador = []
fatiga_perdedor = []
l_racha_w, l_racha_l = [], []

for index, row in df.iterrows():
    tid = row['tourney_id']
    w = row['winner_name']
    l = row['loser_name']
    duracion = row['minutes']
    
    # 1. RECUPERAR FATIGA ACTUAL (Antes de jugar este partido)
    # Si no existe en el diccionario, es su primer partido del torneo -> 0 cansancio
    f_w = fatiga_tracker.get((tid, w), 0)
    f_l = fatiga_tracker.get((tid, l), 0)
    
    fatiga_ganador.append(f_w)
    fatiga_perdedor.append(f_l)
    
    # 2. ACTUALIZAR FATIGA (Para la siguiente ronda)
    # Sumamos lo que acaban de jugar
    fatiga_tracker[(tid, w)] = f_w + duracion
    # Al perdedor también se le suma, aunque ya quede eliminado (por si acaso)
    fatiga_tracker[(tid, l)] = f_l + duracion

    # --- B. MOMENTUM (NUEVO) ---
    # Recuperamos historial reciente (últimos 5)
    hist_w = racha_tracker.get(w, [])
    hist_l = racha_tracker.get(l, [])
    
    # Calculamos el promedio (Win Rate reciente)
    # Si no tiene partidos previos, asumimos 0.5 (neutral)
    mom_w = sum(hist_w) / len(hist_w) if len(hist_w) > 0 else 0.5
    mom_l = sum(hist_l) / len(hist_l) if len(hist_l) > 0 else 0.5
    
    l_racha_w.append(mom_w)
    l_racha_l.append(mom_l)

    # ACTUALIZAMOS HISTORIAL (Después del partido)
    # Al ganador le agregamos un 1
    hist_w.append(1)
    if len(hist_w) > 5: hist_w.pop(0) # Mantenemos solo los últimos 5
    racha_tracker[w] = hist_w
    
    # Al perdedor le agregamos un 0
    hist_l.append(0)
    if len(hist_l) > 5: hist_l.pop(0)
    racha_tracker[l] = hist_l

# Agregamos las columnas al DataFrame
df['winner_fatigue'] = fatiga_ganador
df['loser_fatigue'] = fatiga_perdedor
df['winner_momentum'] = l_racha_w
df['loser_momentum'] = l_racha_l

# -------------------------------------------------------------------------
# LIMPIEZA
# -------------------------------------------------------------------------
# Nos quedamos solo con las columnas que usara el modelo y eliminamos filas con datos faltantes en esas columnas.
cols = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age', 'winner_ht', 'loser_ht', 'surface', 'winner_ioc', 'loser_ioc', 'winner_fatigue', 'loser_fatigue', 'winner_momentum', 'loser_momentum']
df = df.dropna(subset=cols)

# B. Creación del Dataset de Entrenamiento
data_rows = []
df_sample = df.sample(frac=1, random_state=42) # Usar 100% de datos

# Para cada partido, creamos dos filas: una con el ganador como Target 1 y otra con el perdedor como Target 0 
for index, row in df_sample.iterrows():
    surf = row['surface']
    t_ioc = row['tourney_ioc']
    
    # Skills
    skill_w = get_skill(row['winner_name'], surf)
    skill_l = get_skill(row['loser_name'], surf)
    
    # Localía (1 si es local, 0 si no)
    home_w = 1 if row['winner_ioc'] == t_ioc else 0
    home_l = 1 if row['loser_ioc'] == t_ioc else 0
    
    # Diferencia de Localía:
    d_home = home_w - home_l # 1 (J1 Local), -1 (J2 Local), 0 (Neutros)

    # Fatiga 
    # Si d_fatigue es POSITIVO, significa que el ganador estaba MÁS CANSADO que el perdedor
    # La IA debería aprender que valores altos aquí son malos
    d_fatigue = row['winner_fatigue'] - row['loser_fatigue']

    d_momentum = row['winner_momentum'] - row['loser_momentum'] # <-- Nueva Variable

    # Variables base
    d_rank = row['loser_rank'] - row['winner_rank']
    d_age = row['winner_age'] - row['loser_age']
    d_ht = row['winner_ht'] - row['loser_ht']
    d_skill = skill_w - skill_l

    # ESCENARIO 1: Ganador es Target 1
    data_rows.append({
        'diff_rank': d_rank, 'diff_age': d_age, 'diff_ht': d_ht, 
        'diff_skill': d_skill, 'diff_home': d_home,
        'diff_fatigue': d_fatigue, 'diff_momentum': d_momentum,
        'target': 1
    })
    
    # ESCENARIO 2: Ganador es Target 0 (Invertimos todo)
    data_rows.append({
        'diff_rank': -d_rank, 'diff_age': -d_age, 'diff_ht': -d_ht, 
        'diff_skill': -d_skill, 'diff_home': -d_home,
        'diff_fatigue': -d_fatigue, 'diff_momentum': -d_momentum,
        'target': 0
    })

df_train = pd.DataFrame(data_rows)

# 5. ENTRENAMIENTO
print("🤖 Entrenando Modelo Final...")
features = ['diff_rank', 'diff_age', 'diff_ht', 'diff_skill', 'diff_home', 'diff_fatigue', 'diff_momentum']
X = df_train[features]
y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
modelo.fit(X_train, y_train)

acc = accuracy_score(y_test, modelo.predict(X_test))

print("\n" + "="*40)
print(f"🚀 PRECISIÓN V3: {acc:.2%}")
print("="*40)

# Ver importancia de variables (¿Qué tanto le importa la fatiga a la IA?)
importancias = pd.DataFrame({'Variable': features, 'Importancia': modelo.feature_importances_}).sort_values('Importancia', ascending=False)
print("\n📊 QUÉ MIRA LA IA:")
print(importancias)

joblib.dump(modelo, 'modelo_tenis_v4.pkl')