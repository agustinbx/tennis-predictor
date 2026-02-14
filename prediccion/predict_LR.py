import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("🏆 Entrenando y Guardando el Modelo Campeón (Regresión Logística)...")

# --- 1. PREPARACIÓN DE DATOS (Tu lógica de siempre) ---
try:
    df = pd.read_csv("historial_tenis_COMPLETO.csv")
    df['minutes'] = df['minutes'].fillna(100)
    df = df.sort_values(by=['tourney_date', 'tourney_id', 'match_num'])
except:
    print("❌ Error cargando CSV")
    exit()

# SKILL EN SUPERFICIE
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

def get_skill(p, s): return stats_dict.get((p, s), 0.5)

# LOCALÍA
def detectar_pais(nombre):
    t = str(nombre).upper()
    if any(x in t for x in ['MADRID', 'BARCELONA', 'VALENCIA']): return 'ESP'
    if any(x in t for x in ['PARIS', 'ROLAND GARROS']): return 'FRA'
    if any(x in t for x in ['US OPEN', 'INDIAN WELLS', 'MIAMI']): return 'USA'
    if any(x in t for x in ['WIMBLEDON', 'LONDON']): return 'GBR'
    if any(x in t for x in ['AUSTRALIAN', 'MELBOURNE']): return 'AUS'
    if any(x in t for x in ['ROME', 'ROMA']): return 'ITA'
    if any(x in t for x in ['BUENOS AIRES', 'CORDOBA']): return 'ARG'
    return 'NEUTRAL'
df['tourney_ioc'] = df['tourney_name'].apply(detectar_pais)

# FATIGA & MOMENTO DEL JUGADOR (Últimos 5 partidos)
fatiga_tracker = {}
racha_tracker = {}
l_fatiga_w, l_fatiga_l, l_racha_w, l_racha_l = [], [], [], []

for index, row in df.iterrows():
    tid, w, l, dur = row['tourney_id'], row['winner_name'], row['loser_name'], row['minutes']
    
    # Fatiga
    f_w, f_l = fatiga_tracker.get((tid, w), 0), fatiga_tracker.get((tid, l), 0)
    l_fatiga_w.append(f_w); l_fatiga_l.append(f_l)
    fatiga_tracker[(tid, w)], fatiga_tracker[(tid, l)] = f_w + dur, f_l + dur
    
    # Momentum
    hw, hl = racha_tracker.get(w, []), racha_tracker.get(l, [])
    mw = sum(hw)/len(hw) if hw else 0.5
    ml = sum(hl)/len(hl) if hl else 0.5
    l_racha_w.append(mw); l_racha_l.append(ml)
    hw.append(1); hl.append(0)
    if len(hw)>5: hw.pop(0)
    if len(hl)>5: hl.pop(0)
    racha_tracker[w] = hw; racha_tracker[l] = hl

df['winner_fatigue'], df['loser_fatigue'] = l_fatiga_w, l_fatiga_l
df['winner_momentum'], df['loser_momentum'] = l_racha_w, l_racha_l

# -------------------------------------------------------------------------
# CÁLCULO DE HEAD-TO-HEAD (H2H) ⚔️
# -------------------------------------------------------------------------
print("⚔️ Calculando historial entre jugadores (H2H)...")

# Diccionario: {(JugadorA, JugadorB): [VictoriasA, VictoriasB]}
# Ordenamos los nombres alfabéticamente en la clave para que (Nadal, Federer) sea lo mismo que (Federer, Nadal)
h2h_tracker = {} 

l_h2h_w = [] # Ventaja del ganador sobre el perdedor
l_h2h_l = [] # Ventaja del perdedor sobre el ganador (siempre será negativa o cero respecto al otro)

for index, row in df.iterrows():
    w = row['winner_name']
    l = row['loser_name']
    
    # Clave única ordenada alfabéticamente
    p1, p2 = sorted([w, l])
    key = (p1, p2)
    
    # 1. RECUPERAR ESTADO ACTUAL (Antes del partido)
    record = h2h_tracker.get(key, [0, 0]) # [Wins_P1, Wins_P2]
    
    # Identificar quién es P1 y quién es P2 en este match específico
    if w == p1:
        wins_w_prev = record[0]
        wins_l_prev = record[1]
    else:
        wins_w_prev = record[1]
        wins_l_prev = record[0]
    
    # Diferencia H2H: (Victorias Previas del Ganador - Victorias Previas del Perdedor)
    l_h2h_w.append(wins_w_prev - wins_l_prev)
    l_h2h_l.append(wins_l_prev - wins_w_prev)
    
    # 2. ACTUALIZAR PARA EL FUTURO
    if w == p1:
        record[0] += 1
    else:
        record[1] += 1
    h2h_tracker[key] = record

df['winner_h2h'] = l_h2h_w
df['loser_h2h'] = l_h2h_l

cols = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age', 'winner_ht', 'loser_ht', 'surface', 'winner_ioc', 'loser_ioc', 'winner_fatigue', 'loser_fatigue', 'winner_momentum', 'loser_momentum', 'winner_h2h', 'loser_h2h']
df = df.dropna(subset=cols)

# --- 2. CREACIÓN DEL DATASET ---
data_rows = []
df_sample = df.sample(frac=1, random_state=42)

for index, row in df_sample.iterrows():
    surf = row['surface']
    t_ioc = row['tourney_ioc']
    skill_w, skill_l = get_skill(row['winner_name'], surf), get_skill(row['loser_name'], surf)
    h_w, h_l = (1 if row['winner_ioc'] == t_ioc else 0), (1 if row['loser_ioc'] == t_ioc else 0)
    d_h2h = row['winner_h2h'] - row['loser_h2h']
    w_pts = row['winner_rank_points'] if pd.notna(row['winner_rank_points']) else 0
    l_pts = row['loser_rank_points'] if pd.notna(row['loser_rank_points']) else 0
    
    diffs = {
        'diff_rank': row['loser_rank'] - row['winner_rank'],
        'diff_rank_points': w_pts - l_pts,
        'diff_age': row['winner_age'] - row['loser_age'],
        'diff_ht': row['winner_ht'] - row['loser_ht'],
        'diff_skill': skill_w - skill_l,
        'diff_home': h_w - h_l,
        'diff_fatigue': row['winner_fatigue'] - row['loser_fatigue'],
        'diff_h2h': d_h2h,
        'diff_momentum': row['winner_momentum'] - row['loser_momentum']
    }
    
    d1 = diffs.copy(); d1['target'] = 1
    data_rows.append(d1)
    d0 = {k: -v for k, v in diffs.items()}; d0['target'] = 0
    data_rows.append(d0)

df_train = pd.DataFrame(data_rows)

# --- 3. ENTRENAMIENTO, EVALUACION Y GUARDADO ---
# Definimos las columnas (Features) y el Objetivo (Target)
features = ['diff_rank', 'diff_rank_points', 'diff_age', 'diff_ht', 'diff_skill', 'diff_home', 'diff_fatigue', 'diff_momentum', 'diff_h2h']
X = df_train[features]
y = df_train['target']

# A. DIVIDIR DATOS (Train 80% - Test 20%)
# Esto separa los datos para que podamos evaluar el modelo honestamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# B. ESCALAR (Fundamental para Regresión Logística)
scaler = StandardScaler()

# ¡OJO! Ajustamos el escalador SOLO con los datos de entrenamiento (X_train)
# y luego transformamos ambos. Esto evita "contaminar" el test.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Valores de C a probar (regularización inversa: menor C = más regularización)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# C. ENTRENAR EL MODELO
# Modelo base
logistic_model = LogisticRegression(penalty='l2', max_iter=2000, random_state=42)

# Configurar GridSearch
print(f"🔎 Iniciando GridSearch con {len(X_train)} muestras...")
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Entrenar
grid_search.fit(X_train_scaled, y_train)

# Resultados
print("\n" + "="*40)
print(f"✨ Mejores parámetros: {grid_search.best_params_}")
print(f"✨ Mejor exactitud en validación (CV): {grid_search.best_score_:.4f}")
print("="*40)

# D. Evaluar en Test Set (El real)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Precisión Final en Test Set: {acc*100:.2f}%")

# E. Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(cm)

# Intentar graficar si es posible
try:
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Mejor Modelo')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.savefig('matriz_confusion.png')
    print("📊 Gráfico guardado como 'matriz_confusion.png'")
except:
    print("⚠️ No se pudo generar el gráfico (falta entorno gráfico), pero el modelo se entrenó bien.")

# F. Guardar
joblib.dump(best_model, 'modelo_logistico_final.pkl')
joblib.dump(scaler, 'scaler_final.pkl') 
print("\n✅ Archivos guardados: modelo_logistico_final.pkl, scaler_final.pkl")