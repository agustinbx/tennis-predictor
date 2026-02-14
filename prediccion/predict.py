import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
from scipy.stats import randint, uniform

print("🚀 INICIANDO OPTIMIZACIÓN AVANZADA DEL MODELO...")

# 1. CARGAR DATOS
df = pd.read_csv('historial_tenis_COMPLETO.csv')
df = df[df['tourney_date'] > 0].copy()
df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
df = df.sort_values(['tourney_date', 'match_num'])

# =============================================================================
# 🧠 1. ELO RATING (Mantener lo que funciona)
# =============================================================================
print("   -> Recalculando Elo Rating...")

def calcular_elo(df_matches):
    elo_dict = {}
    elo_surf_dict = {}
    w_elo, l_elo = [], []
    w_elo_surf, l_elo_surf = [], []
    
    STARTING_ELO = 1500
    
    for idx, row in df_matches.iterrows():
        w, l, surf = row['winner_name'], row['loser_name'], row['surface']
        
        # Get Elo
        we = elo_dict.get(w, STARTING_ELO)
        le = elo_dict.get(l, STARTING_ELO)
        wes = elo_surf_dict.get((w, surf), STARTING_ELO)
        les = elo_surf_dict.get((l, surf), STARTING_ELO)
        
        w_elo.append(we); l_elo.append(le)
        w_elo_surf.append(wes); l_elo_surf.append(les)
        
        # Calc Prob
        pw = 1 / (1 + 10 ** ((le - we) / 400))
        pws = 1 / (1 + 10 ** ((les - wes) / 400))
        
        # Update
        k = 32
        if 'Grand Slam' in str(row['tourney_level']): k = 50
        elif 'Masters' in str(row['tourney_level']): k = 40
        
        delta = k * (1 - pw)
        delta_s = k * (1 - pws)
        
        elo_dict[w] = we + delta
        elo_dict[l] = le - delta
        elo_surf_dict[(w, surf)] = wes + delta_s
        elo_surf_dict[(l, surf)] = les - delta_s
        
    return w_elo, l_elo, w_elo_surf, l_elo_surf, elo_dict, elo_surf_dict

we, le, wes, les, dict_elo, dict_surf = calcular_elo(df)
df['w_elo'] = we; df['l_elo'] = le
df['w_elo_surf'] = wes; df['l_elo_surf'] = les

# Guardar diccionarios para la App
joblib.dump(dict_elo, 'elo_rating.pkl')
joblib.dump(dict_surf, 'elo_rating_surface.pkl')

# =============================================================================
# 🔋 2. RECUPERAR FATIGA Y PUNTOS DE RANKING
# =============================================================================
print("   -> Calculando Fatiga y H2H...")

fatiga = {}
h2h = {}
w_fat, l_fat = [], []
w_h2h, l_h2h = [], []

for idx, row in df.iterrows():
    tid, w, l, mins = row['tourney_id'], row['winner_name'], row['loser_name'], row['minutes']
    mins = 100 if pd.isna(mins) or mins == 0 else mins # Relleno inteligente
    
    # Fatiga
    fw = fatiga.get((tid, w), 0)
    fl = fatiga.get((tid, l), 0)
    w_fat.append(fw); l_fat.append(fl)
    fatiga[(tid, w)] = fw + mins
    fatiga[(tid, l)] = fl + mins
    
    # H2H
    p1, p2 = sorted([w, l])
    rec = h2h.get((p1, p2), 0)
    if w == p1:
        w_h2h.append(rec); l_h2h.append(-rec)
        h2h[(p1, p2)] = rec + 1
    else:
        w_h2h.append(-rec); l_h2h.append(rec)
        h2h[(p1, p2)] = rec - 1

df['w_fat'] = w_fat; df['l_fat'] = l_fat
df['w_h2h'] = w_h2h; df['l_h2h'] = l_h2h

# =============================================================================
# ⚙️ 3. PREPARAR DATASET
# =============================================================================

# Diferencias
df['diff_elo'] = df['w_elo'] - df['l_elo']
df['diff_elo_surf'] = df['w_elo_surf'] - df['l_elo_surf']
df['diff_rank_pts'] = df['winner_rank_points'] - df['loser_rank_points'] # IMPORTANTE
df['diff_h2h'] = df['w_h2h'] - df['l_h2h']
df['diff_fatigue'] = df['w_fat'] - df['l_fat']
df['diff_age'] = df['winner_age'] - df['loser_age']

features = ['diff_elo', 'diff_elo_surf', 'diff_rank_pts', 'diff_h2h', 'diff_fatigue', 'diff_age']

# Filtro moderno (2012+) para evitar ruido antiguo
df_model = df[df['tourney_date'] >= 20120101].dropna(subset=features).copy()

# Balanceo
X = df_model[features]
y = np.ones(len(X))
X_final = pd.concat([X, -X])
y_final = np.concatenate([y, np.zeros(len(X))])

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# =============================================================================
# 🎛️ 4. TUNING DE HIPERPARÁMETROS (LA CLAVE)
# =============================================================================
print(f"🔎 Buscando la mejor configuración para XGBoost ({len(X_train)} datos)...")

# Definimos el espacio de búsqueda
param_dist = {
    'n_estimators': randint(300, 1000),      # Número de árboles
    'learning_rate': uniform(0.01, 0.1),     # Velocidad de aprendizaje
    'max_depth': randint(3, 7),              # Profundidad (evitar muy profundo)
    'subsample': uniform(0.6, 0.4),          # % de datos por árbol
    'colsample_bytree': uniform(0.6, 0.4),   # % de columnas por árbol
    'gamma': uniform(0, 0.5)                 # Reducción mínima de pérdida
}

xgb_model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)

# RandomizedSearchCV probará 10 combinaciones aleatorias
random_search = RandomizedSearchCV(
    xgb_model, 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=3, 
    scoring='accuracy', 
    verbose=1, 
    n_jobs=-1, # Usar todos los núcleos del CPU
    random_state=42
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
print(f"\n✨ Mejor configuración encontrada: {random_search.best_params_}")

# =============================================================================
# 🏆 5. EVALUACIÓN FINAL
# =============================================================================
preds = best_model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("-" * 40)
print(f"🎯 PRECISIÓN MEJORADA: {acc*100:.2f}%")
print("-" * 40)

# Evaluar confianza alta
# ¿Qué tan bueno es el modelo cuando está SEGURO (>60%)?
probs = best_model.predict_proba(X_test)[:, 1]
high_conf_indices = [i for i, p in enumerate(probs) if p > 0.6 or p < 0.4]
if high_conf_indices:
    y_test_high = y_test[high_conf_indices]
    preds_high = preds[high_conf_indices]
    acc_high = accuracy_score(y_test_high, preds_high)
    print(f"🦁 Precisión en apuestas de Alta Confianza (>60%): {acc_high*100:.2f}%")
    print(f"   (Representa el {len(high_conf_indices)/len(y_test)*100:.1f}% de los partidos)")

# Guardar
scaler = StandardScaler()
scaler.fit(X_final)
joblib.dump(best_model, 'modelo_xgboost_optimizado.pkl')
joblib.dump(scaler, 'scaler_optimizado.pkl')
print("\n✅ Archivos guardados con éxito.")