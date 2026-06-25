#!/usr/bin/env python3
"""
train_ensemble_v3.py — Strict ML implementation of the original 6-model/4-window ensemble.

Maintains the 24-model structure but fixes:
  ✓ Temporal train/val/test split (no meta-learner leakage)
  ✓ Isotonic probability calibration on Val set
  ✓ Median imputation (not fillna(0))
  ✓ Standard Scaler applied properly
  ✓ Curated ~45 features (no bloat)
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# ── Setup paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import sqlite3
from config import CONFIG
from features import build_all_features

DB_PATH = str(CONFIG.paths.db)
V3_DIR = os.path.join(BASE_DIR, 'models_v3')
os.makedirs(V3_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CURATED FEATURE LIST (~45 high-signal features)
# ══════════════════════════════════════════════════════════════════════════════
FEATURES_V3 = [
    'h_starter_era', 'a_starter_era', 'h_starter_whip', 'a_starter_whip',
    'h_starter_strikeout_rate', 'a_starter_strikeout_rate',
    'h_starter_era_l3', 'a_starter_era_l3', 'h_pitcher_form_score', 'a_pitcher_form_score',
    'h_pitcher_quality_idx', 'a_pitcher_quality_idx',
    'h_ops', 'a_ops', 'h_avg', 'a_avg',
    'era_diff', 'whip_diff', 'ops_diff', 'k_rate_diff', 'lineup_power_diff', 'domination_diff',
    'h_starter_xwoba', 'a_starter_xwoba', 'h_lineup_xwoba', 'a_lineup_xwoba',
    'matchup_advantage', 'xwoba_lineup_diff', 'h_starter_barrel', 'a_starter_barrel',
    'park_factor', 'is_night', 'h_rest_days', 'a_rest_days', 'divisional_game',
    'elo_diff', 'streak_diff', 'h_pyth_pct', 'a_pyth_pct',
    'h_home_wpct', 'a_away_wpct', 'run_diff_diff', 'h2h_advantage',
    'h_bp_avail', 'a_bp_avail', 'h_bullpen_era_l3', 'a_bullpen_era_l3',
    'mc_home_prob', 'mc_margin_expected',
]

WINDOWS = {'all': None, 'w50': 750, 'w25': 375, 'w10': 150}

def _make_models():
    return {
        'xgb1': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                    eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1),
        'xgb2': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03,
                    subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0,
                    eval_metric='logloss', verbosity=0, random_state=123, n_jobs=-1),
        'hgb1': HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.04,
                    l2_regularization=1.5, min_samples_leaf=15, random_state=42),
        'hgb2': HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.03,
                    l2_regularization=3.0, min_samples_leaf=20, random_state=123),
        'rf2':  RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=12,
                    class_weight='balanced', random_state=123, n_jobs=-1),
        'mlp':  MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42,
                    early_stopping=True, validation_fraction=0.15, alpha=0.01),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING & PREP
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("═" * 70)
    print(" 📊 LOADING DATA FROM DATABASE")
    print("═" * 70)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM historico_partidos WHERE h_runs_total IS NOT NULL AND a_runs_total IS NOT NULL ORDER BY date", conn)
    pp = pd.read_sql("SELECT * FROM pitcher_performances ORDER BY date", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    pp['date'] = pd.to_datetime(pp['date'], format='mixed')
    print("\n  ⚙️  Building features...")
    df, all_feats, target = build_all_features(df, pp, predict_mode=False)
    return df, target

def prepare_features(df, target, feature_list):
    print("\n" + "═" * 70)
    print(" 🔪 TEMPORAL SPLIT + FEATURE PREP")
    print("═" * 70)
    
    available = [f for f in feature_list if f in df.columns]
    
    # Temporal split
    train_mask = df['date'] < '2025-01-01'     # 2023-2024
    val_mask = (df['date'] >= '2025-01-01') & (df['date'] < '2026-01-01')  # 2025
    test_mask = df['date'] >= '2026-01-01'      # 2026

    X_train_raw = df.loc[train_mask, available].copy()
    X_val_raw = df.loc[val_mask, available].copy()
    X_test_raw = df.loc[test_mask, available].copy()

    y_train = target[train_mask.values]
    y_val = target[val_mask.values]
    y_test = target[test_mask.values]

    # Median imputation (computed on TRAIN only)
    medians = {}
    for col in available:
        med = X_train_raw[col].median()
        medians[col] = float(med) if not pd.isna(med) else 0.0

    X_train_imputed = X_train_raw.fillna(medians).values.astype(np.float32)
    X_val_imputed = X_val_raw.fillna(medians).values.astype(np.float32)
    X_test_imputed = X_test_raw.fillna(medians).values.astype(np.float32)

    # Scaling (computed on TRAIN only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed)
    X_val = scaler.transform(X_val_imputed)
    X_test = scaler.transform(X_test_imputed)

    return X_train, y_train, X_val, y_val, X_test, y_test, available, medians, scaler

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df, target = load_data()
    X_tr, y_tr, X_va, y_va, X_te, y_te, feats, medians, scaler = prepare_features(df, target, FEATURES_V3)
    
    print("\n" + "═" * 70)
    print(" 🏗️  TRAINING BASE MODELS & CALIBRATORS")
    print("═" * 70)

    all_val_preds = []
    
    # Train the 4 windows
    for wname, wsize in WINDOWS.items():
        print(f"\n  🪟  Window: {wname} (size: {wsize or 'ALL'})")
        # Slice the train set
        if wsize and len(X_tr) > wsize:
            X_tr_w = X_tr[-wsize:]
            y_tr_w = y_tr[-wsize:]
        else:
            X_tr_w = X_tr
            y_tr_w = y_tr
            
        models = _make_models()
        
        for mname, model in models.items():
            print(f"      ▶ Training {mname}...")
            model.fit(X_tr_w, y_tr_w)
            
            # Predict on Val set to build Calibrator
            raw_val_probs = model.predict_proba(X_va)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrated_val_probs = calibrator.fit_transform(raw_val_probs, y_va)
            
            # Save
            joblib.dump(model, os.path.join(V3_DIR, f'{wname}_{mname}.pkl'))
            joblib.dump(calibrator, os.path.join(V3_DIR, f'{wname}_{mname}_calibrator.pkl'))
            
            if wname == 'all':
                all_val_preds.append(calibrated_val_probs)

    print("\n" + "═" * 70)
    print(" 🧠 TRAINING META-LEARNER")
    print("═" * 70)
    
    meta_X_va = np.column_stack(all_val_preds)
    meta_learner = LogisticRegression(C=0.1)
    # Fit strictly on Validation Set
    meta_learner.fit(meta_X_va, y_va)
    
    exp_w = np.exp(meta_learner.coef_[0])
    dist_w = exp_w / np.sum(exp_w)
    mkeys = list(_make_models().keys())
    WEIGHTS = {mname: float(dist_w[i]) for i, mname in enumerate(mkeys)}
    
    print(f"  ✅ Stacking weights learned: {WEIGHTS}")
    joblib.dump(meta_learner, os.path.join(V3_DIR, 'meta_learner.pkl'))
    with open(os.path.join(V3_DIR, 'weights.json'), 'w') as f: json.dump(WEIGHTS, f)
    
    # Save artifacts
    joblib.dump(scaler, os.path.join(V3_DIR, 'scaler.pkl'))
    with open(os.path.join(V3_DIR, 'features_v3.json'), 'w') as f: json.dump(feats, f)
    with open(os.path.join(V3_DIR, 'medians_v3.json'), 'w') as f: json.dump(medians, f)
    
    print("\n" + "═" * 70)
    print(" 📈 EVALUATING ON UNSEEN TEST SET (2026)")
    print("═" * 70)
    
    test_preds = []
    for mname in mkeys:
        model = joblib.load(os.path.join(V3_DIR, f'all_{mname}.pkl'))
        calibrator = joblib.load(os.path.join(V3_DIR, f'all_{mname}_calibrator.pkl'))
        raw_test_probs = model.predict_proba(X_te)[:, 1]
        calibrated_test_probs = calibrator.transform(raw_test_probs)
        test_preds.append(calibrated_test_probs)
        
    meta_X_te = np.column_stack(test_preds)
    final_probs = meta_learner.predict_proba(meta_X_te)[:, 1]
    
    acc = accuracy_score(y_te, final_probs >= 0.5)
    loss = log_loss(y_te, final_probs)
    brier = brier_score_loss(y_te, final_probs)
    
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Log Loss:    {loss:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    
    print("\n  ✅ DONE — Models saved to models_v3/")

if __name__ == '__main__':
    main()
