import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

print("🚀 ENTRENANDO EL NUEVO CAMPEÓN (XGBOOST)...")

# --- 1. PREPARACIÓN DE DATOS (Exactamente igual que antes) ---
try:
    df = pd.read_csv("historialTenis.csv")
    df['minutes'] = df['minutes'].fillna(100)
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
    df = df.sort_values(by=['tourney_date', 'match_num'])
except:
    print("❌ Error cargando CSV")
    exit()

# --- VARIABLES ---
print("   -> Generando variables (Skills, H2H, Fatiga, Puntos)...")

# Skills
wins = df.groupby(['winner_name', 'surface']).size().reset_index(name='wins')
losses = df.groupby(['loser_name', 'surface']).size().reset_index(name='losses')
wins.columns = ['player', 'surface', 'wins']
losses.columns = ['player', 'surface', 'losses']
stats = pd.merge(wins, losses, on=['player', 'surface'], how='outer').fillna(0)
stats['total'] = stats['wins'] + stats['losses']
stats = stats[stats['total'] >= 5]
stats['win_rate'] = stats['wins'] / stats['total']
stats_dict = stats.set_index(['player', 'surface'])['win_rate'].to_dict()
joblib.dump(stats_dict, 'stats_superficie_v2.pkl') # Guardamos también los skills

def get_skill(p, s): return stats_dict.get((p, s), 0.5)

# Localía
def detectar_pais(nombre):
    t = str(nombre).upper()
    if 'MADRID' in t or 'BARCELONA' in t: return 'ESP'
    if 'PARIS' in t or 'ROLAND GARROS' in t: return 'FRA'
    if 'US OPEN' in t or 'INDIAN WELLS' in t: return 'USA'
    if 'WIMBLEDON' in t or 'LONDON' in t: return 'GBR'
    if 'AUSTRALIAN' in t or 'MELBOURNE' in t: return 'AUS'
    return 'NEUTRAL'
df['tourney_ioc'] = df['tourney_name'].apply(detectar_pais)

# Fatiga, Momentum, H2H
fatiga_tracker = {}
racha_tracker = {}
h2h_tracker = {}
l_fatiga_w, l_fatiga_l = [], []
l_racha_w, l_racha_l = [], []
l_h2h_w, l_h2h_l = [], []

for index, row in df.iterrows():
    tid, w, l, dur = row['tourney_id'], row['winner_name'], row['loser_name'], row['minutes']
    
    # Fatiga
    f_w = fatiga_tracker.get((tid, w), 0); f_l = fatiga_tracker.get((tid, l), 0)
    l_fatiga_w.append(f_w); l_fatiga_l.append(f_l)
    fatiga_tracker[(tid, w)] = f_w + dur; fatiga_tracker[(tid, l)] = f_l + dur
    
    # Momentum
    hw = racha_tracker.get(w, []); hl = racha_tracker.get(l, [])
    mw = sum(hw)/len(hw) if hw else 0.5; ml = sum(hl)/len(hl) if hl else 0.5
    l_racha_w.append(mw); l_racha_l.append(ml)
    hw.append(1); hl.append(0)
    if len(hw)>5: hw.pop(0); 
    if len(hl)>5: hl.pop(0)
    racha_tracker[w] = hw; racha_tracker[l] = hl
    
    # H2H
    p1, p2 = sorted([w, l])
    key = (p1, p2)
    record = h2h_tracker.get(key, [0, 0])
    if w == p1:
        l_h2h_w.append(record[0] - record[1])
        l_h2h_l.append(record[1] - record[0])
        record[0] += 1
    else:
        l_h2h_w.append(record[1] - record[0])
        l_h2h_l.append(record[0] - record[1])
        record[1] += 1
    h2h_tracker[key] = record

df['winner_fatigue'] = l_fatiga_w; df['loser_fatigue'] = l_fatiga_l
df['winner_momentum'] = l_racha_w; df['loser_momentum'] = l_racha_l
df['winner_h2h'] = l_h2h_w; df['loser_h2h'] = l_h2h_l

# --- DATASET ---
cols = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age', 'winner_ht', 'loser_ht', 'surface', 'winner_ioc', 'loser_ioc', 'winner_fatigue', 'loser_fatigue', 'winner_momentum', 'loser_momentum', 'winner_h2h', 'loser_h2h']
df = df.dropna(subset=cols)

data_rows = []
df_sample = df.sample(frac=1, random_state=42)

for index, row in df_sample.iterrows():
    surf = row['surface']
    t_ioc = row['tourney_ioc']
    skill_w, skill_l = get_skill(row['winner_name'], surf), get_skill(row['loser_name'], surf)
    h_w = 1 if row['winner_ioc'] == t_ioc else 0
    h_l = 1 if row['loser_ioc'] == t_ioc else 0
    
    pts_w = row['winner_rank_points'] if pd.notna(row['winner_rank_points']) else 0
    pts_l = row['loser_rank_points'] if pd.notna(row['loser_rank_points']) else 0
    
    diffs = {
        'diff_rank': row['loser_rank'] - row['winner_rank'],
        'diff_rank_points': pts_w - pts_l,
        'diff_age': row['winner_age'] - row['loser_age'],
        'diff_ht': row['winner_ht'] - row['loser_ht'],
        'diff_skill': skill_w - skill_l,
        'diff_home': h_w - h_l,
        'diff_fatigue': row['winner_fatigue'] - row['loser_fatigue'],
        'diff_h2h': row['winner_h2h'] - row['loser_h2h'],
        'diff_momentum': row['winner_momentum'] - row['loser_momentum']
    }
    
    d1 = diffs.copy(); d1['target'] = 1
    data_rows.append(d1)
    d0 = {k: -v for k, v in diffs.items()}; d0['target'] = 0
    data_rows.append(d0)

df_train = pd.DataFrame(data_rows)

# --- 2. ENTRENAMIENTO XGBOOST ---
features = ['diff_rank', 'diff_rank_points', 'diff_age', 'diff_ht', 'diff_skill', 'diff_home', 'diff_fatigue', 'diff_momentum', 'diff_h2h']
X = df_train[features]
y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar (Es bueno mantenerlo por consistencia)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"🌲 Entrenando XGBoost con {len(X_train)} datos...")
# Usamos los parámetros que dieron 72.36% en tu comparación
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)

# Resultados
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("-" * 40)
print(f"🏆 PRECISIÓN FINAL (TEST SET): {acc*100:.2f}%")
print("-" * 40)

# Guardar con NOMBRE NUEVO
joblib.dump(model, 'modelo_xgboost_final.pkl')
joblib.dump(scaler, 'scaler_final.pkl') # Sobrescribimos el scaler para que coincida con este modelo
print("✅ Archivos guardados: 'modelo_xgboost_final.pkl' y 'scaler_final.pkl'")