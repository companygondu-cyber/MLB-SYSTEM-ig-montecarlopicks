#!/usr/bin/env python3
"""
train_props.py — Retraining engine for MLB Props (K, H, TB, HR, R)
Loads data from SQLite, extracts features, fits XGBoost models, and saves them.
"""
import os
import sys
import sqlite3
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from config import CONFIG
from models.props import PropsAnalyzerV3, FEAT_NAMES
from models.k import KAnalyzer

DB_PATH = str(CONFIG.paths.db)
K_MODEL_PATH = str(CONFIG.paths.k_props / 'k_xgb.pkl')
K_FEATS_PATH = str(CONFIG.paths.k_props / 'features.json')
SPEC_DIR = str(CONFIG.paths.batter_props / 'specialists')

def train_pitcher_k():
    print("⏳ Entrenando modelo de Ponches (K)...")
    conn = sqlite3.connect(DB_PATH)
    analyzer = KAnalyzer(conn)
    
    # Obtener aperturas de pitchers abridores recientes (últimas 3000)
    # Buscamos game_pk, date, player_id, team_id, y los K reales
    starters = conn.execute('''
        SELECT game_pk, date, player_id, team_id, k, runs
        FROM pitcher_performances
        WHERE role = "starter" AND date >= "2025-04-01"
        ORDER BY date DESC LIMIT 3000
    ''').fetchall()
    
    # Para cada juego, buscar el equipo rival (opp_team) y el estadio (home_park)
    game_details = {}
    for gpk, gd, ht, at, venue in conn.execute('SELECT game_pk, date, home_team, away_team, venue FROM historico_partidos WHERE date >= "2025-01-01"').fetchall():
        game_details[gpk] = (ht, at, venue)
        
    X_train = []
    y_train = []
    
    for gpk, date, pid, team_id, actual_k, runs in starters:
        if gpk not in game_details: continue
        ht, at, venue = game_details[gpk]
        opp_team = at if team_id == analyzer.team_ids.get(ht) else ht
        home_park = venue
        
        res = analyzer.get_features(pid, opp_team, home_park, date)
        if res:
            features_list, _ = res
            X_train.append(features_list)
            y_train.append(actual_k)
            
    conn.close()
    
    if len(X_train) < 100:
        print("⚠️ No hay suficientes datos históricos para entrenar el modelo K.")
        return
        
    X = np.nan_to_num(np.array(X_train), nan=0.0)
    y = np.array(y_train)
    
    print(f"   ├─ Muestras de entrenamiento: {len(X)}")
    model = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.04, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Guardar modelo y features
    os.makedirs(os.path.dirname(K_MODEL_PATH), exist_ok=True)
    joblib.dump(model, K_MODEL_PATH)
    
    feat_cols = ["k9_l3", "k9_l10", "k9_szn", "k9_car", "xwoba", "spin", "ext", "ev", "barrel", "strike_pct", "ip_avg", "chase", "park", "rest"]
    with open(K_FEATS_PATH, 'w') as f:
        json.dump(feat_cols, f)
    print("   └─ ✅ Modelo K reentrenado y guardado con éxito.")

def train_batter_specialists():
    print("⏳ Entrenando especialistas de bateo (H, TB, HR, R)...")
    analyzer = PropsAnalyzerV3()
    conn = analyzer.conn
    
    # Obtener bateos recientes (últimas 5000 apariciones registradas)
    # Queremos game_pk, date, player_id, team_id, hits, runs, hr, tb
    # Para evitar que tarde demasiado, filtramos por jugadores regulares
    batters = conn.execute('''
        SELECT bp.game_pk, bp.date, bp.player_id, bp.team_id,
               pgs.hits, pgs.runs, pgs.hr, pgs.tb
        FROM batter_performances bp
        JOIN player_game_stats pgs ON bp.game_pk = pgs.game_pk AND bp.player_id = pgs.player_id
        WHERE bp.date >= "2025-04-01"
        ORDER BY bp.date DESC
    ''').fetchall()
    
    game_details = {}
    for gpk, gd, ht, at, venue in conn.execute('SELECT game_pk, date, home_team, away_team, venue FROM historico_partidos WHERE date >= "2025-01-01"').fetchall():
        game_details[gpk] = (ht, at, venue)
        
    # Obtener el lanzador abridor rival para cada juego para cruzar con batter_features
    pitchers_opp = {}
    for gpk, pid, team_id in conn.execute('SELECT game_pk, player_id, team_id FROM pitcher_performances WHERE role="starter"').fetchall():
        pitchers_opp[(gpk, team_id)] = pid

    X_train = []
    y_h = []
    y_tb = []
    y_hr = []
    y_r = []
    
    for gpk, date, pid, team_id, hits, runs, hr, tb in batters:
        if gpk not in game_details: continue
        ht, at, venue = game_details[gpk]
        is_home = (team_id == analyzer.team_ids.get(ht))
        opp_team = at if is_home else ht
        opp_team_id = analyzer.team_ids.get(opp_team)
        
        opp_pitcher_id = pitchers_opp.get((gpk, opp_team_id), 0)
        if not opp_pitcher_id: continue
        
        feats = analyzer.batter_features(pid, opp_pitcher_id, opp_team, venue, is_home, date)
        X_train.append(feats)
        y_h.append(1 if hits > 0 else 0)
        y_tb.append(1 if tb >= 1.5 else 0)
        y_hr.append(1 if hr > 0 else 0)
        y_r.append(1 if runs > 0 else 0)
        
    if len(X_train) < 100:
        print("⚠️ No hay suficientes datos históricos para entrenar los modelos de bateo.")
        return
        
    X = np.nan_to_num(np.array(X_train), nan=0.0)
    print(f"   ├─ Muestras de entrenamiento: {len(X)}")
    
    os.makedirs(SPEC_DIR, exist_ok=True)
    
    targets = {
        'H': (np.array(y_h), 'omega_h_v3.pkl'),
        'TB': (np.array(y_tb), 'omega_tb_v3.pkl'),
        'HR': (np.array(y_hr), 'omega_hr_v3.pkl'),
        'R': (np.array(y_r), 'omega_r_v3.pkl')
    }
    
    for prop, (y, filename) in targets.items():
        print(f"   ├─ Entrenando especialista de {prop}...")
        # Clasificador balanceado
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='logloss')
        model.fit(X, y)
        joblib.dump(model, os.path.join(SPEC_DIR, filename))
        
    print("   └─ ✅ Especialistas de bateo reentrenados y guardados.")

def run_retraining():
    print("======================================================================")
    print(" 🧠 INICIANDO REENTRENAMIENTO DINÁMICO DE PROPS (MLB)")
    print("======================================================================")
    train_pitcher_k()
    train_batter_specialists()
    print("======================================================================")

if __name__ == '__main__':
    run_retraining()
