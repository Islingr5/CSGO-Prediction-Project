import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import joblib 
import warnings
import sys
import os

# =============================================================================
# SETTINGS & CONFIGURATION
# =============================================================================
warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None

print(" [SYSTEM] CS:GO PREDICTOR V11 DEMO STARTING...")

# =============================================================================
# 1. DATA LOADING
# =============================================================================
try:
    df_results = pd.read_csv('results.csv')
    df_players = pd.read_csv('players.csv')
    df_eco = pd.read_csv('economy.csv', low_memory=False) 
except:
    print("ERROR: csv files not found! Please check your directory.")
    sys.exit()

# Function to clean column names (remove spaces, lowercase)
def clean_col_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

df_results = clean_col_names(df_results)
df_players = clean_col_names(df_players)
df_eco = clean_col_names(df_eco)

# Standardize map column name
map_col = '_map' if '_map' in df_results.columns else 'map'
df_results.rename(columns={map_col: '_map'}, inplace=True)

# Drop duplicate or unnecessary columns from economy dataframe to avoid merge conflicts
cols_drop = ['date', 'team_1', 'team_2', '_map', 'event_id', 'best_of']
for c in cols_drop:
    if c in df_eco.columns: df_eco.drop(columns=[c], inplace=True)

# Convert date columns to datetime objects
df_results['date'] = pd.to_datetime(df_results['date'])
df_players['date'] = pd.to_datetime(df_players['date'])
if 'team_name' not in df_players.columns and 'team' in df_players.columns:
    df_players.rename(columns={'team': 'team_name'}, inplace=True)

# Function to normalize team names (handling case sensitivity)
def normalize_names(series):
    return series.str.lower().str.strip().str.replace(' ', '_')

df_results['team_1'] = normalize_names(df_results['team_1'])
df_results['team_2'] = normalize_names(df_results['team_2'])
df_players['team_name'] = normalize_names(df_players['team_name'])

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print(" [PROCESS] Processing data & engineering features...")

# --- Pistol Round Analysis ---
# Merging economy data to determine who won pistol rounds (Round 1 and 16)
df_eco_m = pd.merge(df_eco, df_results[['match_id', 'date', 'team_1', 'team_2']], on='match_id', how='left')
df_eco_m.dropna(subset=['date'], inplace=True)
p_data = []
for i, r in df_eco_m.iterrows():
    try:
        # Check 1st Round Winner
        if pd.notna(r.get('1_winner')):
            w = int(float(r['1_winner']))
            p_data.extend([{'date':r['date'], 'team':r['team_1'], 'pw':1 if w==1 else 0}, {'date':r['date'], 'team':r['team_2'], 'pw':1 if w==2 else 0}])
        # Check 16th Round Winner
        if pd.notna(r.get('16_winner')):
            w = int(float(r['16_winner']))
            p_data.extend([{'date':r['date'], 'team':r['team_1'], 'pw':1 if w==1 else 0}, {'date':r['date'], 'team':r['team_2'], 'pw':1 if w==2 else 0}])
    except: continue

if p_data:
    df_p = pd.DataFrame(p_data).sort_values(by=['team', 'date'])
    # Calculate rolling average of pistol win rate (Last 20 matches)
    df_p['p_wr'] = df_p.groupby('team')['pw'].transform(lambda x: x.rolling(20, min_periods=5).mean().shift(1)).fillna(0.5)
    p_stats = df_p.groupby(['date', 'team'])['p_wr'].mean().reset_index()
else: p_stats = pd.DataFrame(columns=['date', 'team', 'p_wr'])

# --- Rank & ELO Calculation ---
df_results['rank_1'] = pd.to_numeric(df_results['rank_1'], errors='coerce').fillna(300)
df_results['rank_2'] = pd.to_numeric(df_results['rank_2'], errors='coerce').fillna(300)

tr = {} 
df_results.sort_values(by='date', inplace=True)

# Custom ELO Algorithm
def elo(t1, t2, w, k=30):
    r1, r2 = tr.get(t1, 1500), tr.get(t2, 1500)
    e1 = 1/(1+10**((r2-r1)/400)); e2 = 1/(1+10**((r1-r2)/400))
    tr[t1], tr[t2] = r1+k*((1 if w==1 else 0)-e1), r2+k*((1 if w==2 else 0)-e2)
    return r1, r2

el = [elo(r['team_1'], r['team_2'], r['match_winner']) for i, r in df_results.iterrows()]
df_results['t1_elo'] = [x[0] for x in el]; df_results['t2_elo'] = [x[1] for x in el]

# --- Team Momentum & Player Impact Stats ---
t1r = df_results[['date','team_1','match_winner']].rename(columns={'team_1':'team'}); t1r['w']=(t1r['match_winner']==1).astype(int)
t2r = df_results[['date','team_2','match_winner']].rename(columns={'team_2':'team'}); t2r['w']=(t2r['match_winner']==2).astype(int)
th = pd.concat([t1r, t2r]).sort_values(by=['team','date'])
# Rolling Win Rate (Last 5 games)
th['wr'] = th.groupby('team')['w'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1)).fillna(0.5)
t_wr = th.groupby(['date','team'])['wr'].mean().reset_index()

# Player Impact Rating Calculation
df_players.fillna(0, inplace=True)
df_players['imp'] = (df_players['kills']*1.2 + df_players['assists']*0.3 + df_players['fkdiff']*0.7 + df_players['kast']*0.05 + df_players['adr']*0.01 - df_players['deaths']*0.5)
df_players.sort_values(by=['player_name','date'], inplace=True)
# Rolling average of player impact
df_players['avg'] = df_players.groupby('player_name')['imp'].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1)).fillna(0)
t_stats = df_players.groupby(['match_id','team_name'])['avg'].mean().reset_index()

# --- Data Merging ---
print(" [MERGE] Merging datasets...")
df = pd.merge(df_results, t_stats, left_on=['match_id','team_1'], right_on=['match_id','team_name'], how='left').rename(columns={'avg':'t1_p'}).drop(columns=['team_name'])
df = pd.merge(df, t_stats, left_on=['match_id','team_2'], right_on=['match_id','team_name'], how='left').rename(columns={'avg':'t2_p'}).drop(columns=['team_name'])
df = pd.merge(df, t_wr, left_on=['date','team_1'], right_on=['date','team'], how='left').rename(columns={'wr':'t1_wr'}).drop(columns=['team'])
df = pd.merge(df, t_wr, left_on=['date','team_2'], right_on=['date','team'], how='left').rename(columns={'wr':'t2_wr'}).drop(columns=['team'])

if not p_stats.empty:
    df = pd.merge(df, p_stats, left_on=['date','team_1'], right_on=['date','team'], how='left').rename(columns={'p_wr':'t1_pis'}).drop(columns=['team'])
    df = pd.merge(df, p_stats, left_on=['date','team_2'], right_on=['date','team'], how='left').rename(columns={'p_wr':'t2_pis'}).drop(columns=['team'])
else: df['t1_pis']=0.5; df['t2_pis']=0.5

df.dropna(subset=['t1_p','t2_p'], inplace=True)
df.fillna(0.5, inplace=True)

# Calculating Differential Features
df['elo_d'] = df['t1_elo'] - df['t2_elo']
df['rank_d'] = df['rank_2'] - df['rank_1']
df['perf_d'] = df['t1_p'] - df['t2_p']
df['wr_d'] = df['t1_wr'] - df['t2_wr']
df['pis_d'] = df['t1_pis'] - df['t2_pis']
df['elo_x_wr'] = df['elo_d'] * df['wr_d']
df['rank_x_perf'] = df['rank_d'] * df['perf_d']

# One Hot Encoding for Maps
df = pd.get_dummies(df, columns=['_map'], prefix='map')
df.columns = df.columns.str.lower()

# =============================================================================
# 3. MODEL TRAINING (DEMO - NO TRAINING)
# =============================================================================
print(" [SYSTEM] Loading pre-trained model (Presentation Mode)...")

# Upload the model from file
if os.path.exists('csgo_v11_model.pkl'):
    best_model = joblib.load('csgo_v11_model.pkl')
    print(" [OK] Model successfully loaded: csgo_v11_model.pkl")
else:
    print(" [ERROR] 'csgo_v11_model.pkl' not found! Please run the training script first.")
    sys.exit()

cols = ['t1_elo', 't2_elo', 'elo_d', 'rank_1', 'rank_2', 'rank_d', 
        't1_wr', 't2_wr', 'wr_d', 't1_p', 't2_p', 'perf_d', 
        't1_pis', 't2_pis', 'pis_d', 'elo_x_wr', 'rank_x_perf']

map_cols = [c for c in df.columns if c.startswith('map_') and 'win' not in c and 'result' not in c]
cols += map_cols

display_maps = sorted([c.replace('map_', '').title() for c in map_cols])

# =============================================================================
# 4. TIME TRAVELER MODE (SAFE & INTERACTIVE)
# =============================================================================
full_history = df.sort_values(by='date').copy()
all_teams = sorted(list(set(full_history['team_1'].unique()) | set(full_history['team_2'].unique())))

def print_teams_paginated():
    chunk_size = 100 
    total = len(all_teams)
    total_pages = (total // chunk_size) + (1 if total % chunk_size > 0 else 0)
    
    print(f"\n--- TOTAL {total} TEAMS FOUND ({total_pages} Pages) ---\n")
    
    for i in range(0, total, chunk_size):
        chunk = all_teams[i:i + chunk_size]
        current_page = (i // chunk_size) + 1
        
        print(", ".join(chunk))
        print(f"\n--- Page {current_page} / {total_pages} ---")
        
        if i + chunk_size < total:
            choice = input("Press [ENTER] for next page, 'q' to quit list: ")
            if choice.lower() == 'q': break
        else: print("--- End of List ---")

def predict_match_timetravel(t1_name, t2_name, map_name):
    t1 = t1_name.strip().lower().replace(' ', '_')
    t2 = t2_name.strip().lower().replace(' ', '_')
    sel_map = map_name.strip().lower()

    # Check if teams exist
    if t1 not in all_teams:
        print(f" [ERROR] Team '{t1_name}' not found. Type 'list' to see all teams.")
        if "navi" in t1: print(" -> Hint: try 'natus_vincere'.")
        return
    if t2 not in all_teams:
        print(f" [ERROR] Team '{t2_name}' not found.")
        if "navi" in t2: print(" -> Hint: try 'natus_vincere'.")
        return

    # LOGIC: Find the LAST MATCH between these two teams
    mask = ((full_history['team_1'] == t1) & (full_history['team_2'] == t2)) | \
           ((full_history['team_1'] == t2) & (full_history['team_2'] == t1))
    
    matches = full_history[mask]
    
    if matches.empty:
        print(f" [!] These teams have never met in the dataset.")
        
        # Fallback: Use their latest individual stats
        t1_hist = full_history[full_history['team_1'] == t1]
        t2_hist = full_history[full_history['team_1'] == t2]
        
        if t1_hist.empty or t2_hist.empty:
             print(" [ERROR] One of the teams lacks sufficient match history.")
             return

        row1 = t1_hist.iloc[-1]
        row2 = t2_hist.iloc[-1]
        sim_date = "2020 (LATEST FORM)"
        real_outcome = "No Match Found"
        
        features = {}
        features['t1_elo'] = row1['t1_elo']; features['t2_elo'] = row2['t1_elo']
        features['elo_d'] = row1['t1_elo'] - row2['t1_elo']
        features['rank_1'] = row1['rank_1']; features['rank_2'] = row2['rank_1']
        features['rank_d'] = row2['rank_1'] - row1['rank_1']
        features['t1_wr'] = row1['t1_wr']; features['t2_wr'] = row2['t1_wr']
        features['wr_d'] = row1['t1_wr'] - row2['t1_wr']
        features['t1_p'] = row1['t1_p']; features['t2_p'] = row2['t1_p']
        features['perf_d'] = row1['t1_p'] - row2['t1_p']
        features['t1_pis'] = row1['t1_pis']; features['t2_pis'] = row2['t1_pis']
        features['pis_d'] = row1['t1_pis'] - row2['t1_pis']
        features['elo_x_wr'] = features['elo_d'] * features['wr_d']
        features['rank_x_perf'] = features['rank_d'] * features['perf_d']
        for c in map_cols: features[c] = 0

    else:
        # Match found! Time travel to that date.
        last_match = matches.iloc[-1]
        sim_date = last_match['date'].strftime('%Y-%m-%d')
        
        if last_match['team_1'] == t1:
            input_row = last_match.copy()
        else: 
            # Swap logic if T1 was on the right side
            input_row = last_match.copy()
            swap_map = {
                't1_elo': 't2_elo', 't2_elo': 't1_elo',
                'rank_1': 'rank_2', 'rank_2': 'rank_1',
                't1_wr': 't2_wr', 't2_wr': 't1_wr',
                't1_p': 't2_p', 't2_p': 't1_p',
                't1_pis': 't2_pis', 't2_pis': 't1_pis'
            }
            input_row.rename(index=swap_map, inplace=True)
            for c in ['elo_d', 'rank_d', 'perf_d', 'wr_d', 'pis_d', 'elo_x_wr', 'rank_x_perf']:
                input_row[c] = -input_row[c]
            
        w_code = last_match['match_winner']
        real_winner_name = last_match['team_1'] if w_code == 1 else last_match['team_2']
        real_outcome = f"{real_winner_name.upper()} won."
        features = input_row.to_dict() 

    # Map Configuration
    map_key = f"map_{sel_map}"
    
    # Reset all maps to 0
    for c in map_cols: features[c] = 0
    
    # Set selected map
    if map_key in map_cols:
        features[map_key] = 1
    else:
        if sel_map != "": 
             print(f" [WARNING] Map '{sel_map}' not found. Defaulting to Mirage.")
        if 'map_mirage' in map_cols: features['map_mirage'] = 1
        sel_map = "mirage"

    input_df = pd.DataFrame([features])
    input_df = input_df[cols] # Ensure column order
    
    # PREDICTION
    prob = best_model.predict_proba(input_df)[0]
    winner = best_model.predict(input_df)[0]
    
    w_team = t1_name if winner == 0 else t2_name
    conf = prob[0] if winner == 0 else prob[1]
    
    risk_msg = "✅  RELIABLE"
    if 0.50 <= conf < 0.60: risk_msg = "⚠️  VERY HIGH RISK (Coin Flip)"
    elif 0.60 <= conf < 0.70: risk_msg = "⚠️  MEDIUM RISK"
    
    print(f"\n ⚔️  {t1_name.upper()} vs {t2_name.upper()} (Map: {sel_map.upper() if sel_map else 'MIRAGE'})")
    print(f" 📅  DATE: {sim_date}")
    print("-" * 40)
    print(f" 🤖  PREDICTION: {w_team.upper()} (%{conf*100:.1f})")
    print(f" 📊  RISK: {risk_msg}")
    print(f" 🕵️  REAL RESULT: {real_outcome}")
    print("-" * 40)

print("\n" + "="*60)
print("      CS:GO ORACLE V11 (FINAL)      ")
print("="*60)
print("Hint: Type 'list' to see all teams.")
print("Hint: Type 'maplist' to see all maps.")
print("Hint: Type 'q' to quit.\n")

while True:
    t1 = input(">> 1. Team (or 'list', 'maplist'): ")
    
    if t1 == 'q': break
    
    # --- TEAM LIST ---
    if t1 == 'list': 
        print_teams_paginated()
        continue
        
    # --- MAP LIST ---
    if t1 == 'maplist':
        print("\n--- AVAILABLE MAPS ---")
        print(", ".join(display_maps))
        print("-" * 30 + "\n")
        continue
        
    t2 = input(">> 2. Team: ")
    if t2 == 'q': break
    
    mp = input(">> Map: ")
    predict_match_timetravel(t1, t2, mp)
