import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("🚀 TEST DE NUEVAS VARIABLES (SIN LEAKAGE)...")

df = pd.read_csv("historialTenis.csv", low_memory=False)
df['minutes'] = df['minutes'].fillna(100)
df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
df = df.sort_values(by=['tourney_date', 'match_num'])

cols_clutch = ['w_bpSaved', 'w_bpFaced', 'w_SvGms', 'l_bpSaved', 'l_bpFaced', 'l_SvGms']
for c in cols_clutch:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# Trackers (no leakage)
fatiga_tracker = {}
racha_tracker = {}
h2h_tracker = {}
elo_tracker = {}
clutch_tracker = {}
surface_tracker = {}

data_rows = []

def get_clutch_score(stats_list):
    bp_s, bp_f, sv_h, sv_g = stats_list
    bp_rate = bp_s / bp_f if bp_f > 0 else 0.5
    sv_rate = sv_h / sv_g if sv_g > 0 else 0.5
    return (bp_rate + sv_rate) / 2.0

for index, row in df.iterrows():
    tid, w, l, dur = row['tourney_id'], row['winner_name'], row['loser_name'], row['minutes']
    surf = row['surface']
    
    # --- Skill Cumulative ---
    key_w = (w, surf)
    key_l = (l, surf)
    rec_w = surface_tracker.get(key_w, [0, 0])
    rec_l = surface_tracker.get(key_l, [0, 0])
    
    skill_w = rec_w[0] / (rec_w[0]+rec_w[1]) if (rec_w[0]+rec_w[1]) >= 5 else 0.5
    skill_l = rec_l[0] / (rec_l[0]+rec_l[1]) if (rec_l[0]+rec_l[1]) >= 5 else 0.5
    
    rec_w[0] += 1
    rec_l[1] += 1
    surface_tracker[key_w] = rec_w
    surface_tracker[key_l] = rec_l
    
    # ELO
    elo_w_current = elo_tracker.get(w, 1500)
    elo_l_current = elo_tracker.get(l, 1500)
    E_w = 1 / (1 + 10 ** ((elo_l_current - elo_w_current) / 400))
    E_l = 1 / (1 + 10 ** ((elo_w_current - elo_l_current) / 400))
    elo_tracker[w] = elo_w_current + 32 * (1 - E_w)
    elo_tracker[l] = elo_l_current + 32 * (0 - E_l)
    
    # Fatiga
    f_w = fatiga_tracker.get((tid, w), 0)
    f_l = fatiga_tracker.get((tid, l), 0)
    fatiga_tracker[(tid, w)] = f_w + dur
    fatiga_tracker[(tid, l)] = f_l + dur
    
    # Momentum
    hw = racha_tracker.get(w, []); hl = racha_tracker.get(l, [])
    mw = sum(hw)/len(hw) if hw else 0.5; ml = sum(hl)/len(hl) if hl else 0.5
    hw.append(1); hl.append(0)
    if len(hw)>5: hw.pop(0)
    if len(hl)>5: hl.pop(0)
    racha_tracker[w] = hw; racha_tracker[l] = hl
    
    # Clutch
    c_w_stats = clutch_tracker.get(w, [0,0,0,0])
    c_l_stats = clutch_tracker.get(l, [0,0,0,0])
    score_w_c = get_clutch_score(c_w_stats)
    score_l_c = get_clutch_score(c_l_stats)
    
    if 'w_bpSaved' in df.columns:
        c_w_stats[0] += row['w_bpSaved']; c_w_stats[1] += row['w_bpFaced']
        c_l_stats[0] += row['l_bpSaved']; c_l_stats[1] += row['l_bpFaced']
        w_broken = row['w_bpFaced'] - row['w_bpSaved']
        l_broken = row['l_bpFaced'] - row['l_bpSaved']
        c_w_stats[2] += (row['w_SvGms'] - w_broken); c_w_stats[3] += row['w_SvGms']
        c_l_stats[2] += (row['l_SvGms'] - l_broken); c_l_stats[3] += row['l_SvGms']
    clutch_tracker[w] = c_w_stats
    clutch_tracker[l] = c_l_stats
    
    # H2H
    p1, p2 = sorted([w, l])
    key_h2h = (p1, p2)
    record = h2h_tracker.get(key_h2h, [0, 0])
    if w == p1:
        h2h_w = record[0] - record[1]
        h2h_l = record[1] - record[0]
        record[0] += 1
    else:
        h2h_w = record[1] - record[0]
        h2h_l = record[0] - record[1]
        record[1] += 1
    h2h_tracker[key_h2h] = record
    
    # Validations for points/rank:
    r_w = row['winner_rank'] if pd.notna(row['winner_rank']) else 9999
    r_l = row['loser_rank'] if pd.notna(row['loser_rank']) else 9999
    p_w = row['winner_rank_points'] if pd.notna(row['winner_rank_points']) else 0
    p_l = row['loser_rank_points'] if pd.notna(row['loser_rank_points']) else 0
    age_w = row['winner_age'] if pd.notna(row['winner_age']) else 25
    age_l = row['loser_age'] if pd.notna(row['loser_age']) else 25
    ht_w = row['winner_ht'] if pd.notna(row['winner_ht']) else 185
    ht_l = row['loser_ht'] if pd.notna(row['loser_ht']) else 185
    
    diffs = {
        'diff_elo': elo_w_current - elo_l_current,
        'diff_rank': r_l - r_w,  # Lower rank is better
        'diff_points': p_w - p_l,
        'diff_clutch': score_w_c - score_l_c,
        'diff_age': age_w - age_l,
        'diff_ht': ht_w - ht_l,
        'diff_skill': skill_w - skill_l,
        'diff_fatigue': f_w - f_l,
        'diff_momentum': mw - ml,
        'diff_h2h': h2h_w - h2h_l
    }
    
    data_rows.append({**diffs, 'target': 1})
    data_rows.append({k: -v for k, v in diffs.items()})
    data_rows[-1]['target'] = 0

df_train = pd.DataFrame(data_rows).dropna()
features = ['diff_elo', 'diff_rank', 'diff_points', 'diff_clutch', 'diff_age', 'diff_ht', 'diff_skill', 'diff_fatigue', 'diff_momentum', 'diff_h2h']
X = df_train[features]
y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

acc = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"PRECISION: {acc*100:.2f}%")
for i, f in enumerate(features):
    print(f"  {f}: {model.feature_importances_[i]*100:.2f}%")
