#!/usr/bin/env python3
"""
train_xgb_v2.py — Proper single XGBoost model for MLB game prediction.

Fixes every flaw in the original ensemble:
  ✓ Temporal train/val/test split (no leakage)
  ✓ Optuna hyperparameter optimization
  ✓ Monotonic constraints (ERA up = bad, OPS up = good)
  ✓ Early stopping on validation log_loss
  ✓ Isotonic probability calibration
  ✓ Median imputation (not fillna(0))
  ✓ Curated ~45 features (no bloat)
  ✓ Full metrics report + feature importance
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# ── Setup paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    classification_report, confusion_matrix
)
import sqlite3

from config import CONFIG
from features import build_all_features

DB_PATH = str(CONFIG.paths.db)
V2_DIR = os.path.join(BASE_DIR, 'models_v2')
os.makedirs(V2_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CURATED FEATURE LIST (~45 high-signal features)
# ══════════════════════════════════════════════════════════════════════════════
FEATURES_V2 = [
    # ── Pitcher quality (6) ──
    'h_starter_era', 'a_starter_era',
    'h_starter_whip', 'a_starter_whip',
    'h_starter_strikeout_rate', 'a_starter_strikeout_rate',

    # ── Pitcher form (6) ──
    'h_starter_era_l3', 'a_starter_era_l3',
    'h_pitcher_form_score', 'a_pitcher_form_score',
    'h_pitcher_quality_idx', 'a_pitcher_quality_idx',

    # ── Team offense (4) ──
    'h_ops', 'a_ops',
    'h_avg', 'a_avg',

    # ── Key differentials (6) ──
    'era_diff', 'whip_diff', 'ops_diff',
    'k_rate_diff', 'lineup_power_diff', 'domination_diff',

    # ── Savant advanced (8) ──
    'h_starter_xwoba', 'a_starter_xwoba',
    'h_lineup_xwoba', 'a_lineup_xwoba',
    'matchup_advantage', 'xwoba_lineup_diff',
    'h_starter_barrel', 'a_starter_barrel',

    # ── Context (5) ──
    'park_factor', 'is_night',
    'h_rest_days', 'a_rest_days',
    'divisional_game',

    # ── Momentum / Strength (8) ──
    'elo_diff', 'streak_diff',
    'h_pyth_pct', 'a_pyth_pct',
    'h_home_wpct', 'a_away_wpct',
    'run_diff_diff', 'h2h_advantage',

    # ── Bullpen (4) ──
    'h_bp_avail', 'a_bp_avail',
    'h_bullpen_era_l3', 'a_bullpen_era_l3',

    # ── Monte Carlo (2) ──
    'mc_home_prob', 'mc_margin_expected',
]

# ══════════════════════════════════════════════════════════════════════════════
# 2. MONOTONIC CONSTRAINTS
#    XGBoost convention: +1 = higher value → higher P(home_win)
#                        -1 = higher value → lower P(home_win)
#                         0 = unconstrained
# ══════════════════════════════════════════════════════════════════════════════
MONOTONIC_MAP = {
    # Higher home ERA → worse for home
    'h_starter_era': -1, 'a_starter_era': 1,
    'h_starter_whip': -1, 'a_starter_whip': 1,
    # Higher home K rate → better for home (more strikeouts)
    'h_starter_strikeout_rate': 1, 'a_starter_strikeout_rate': -1,
    # Lower ERA L3 → better for home
    'h_starter_era_l3': -1, 'a_starter_era_l3': 1,
    # Higher form → better
    'h_pitcher_form_score': 1, 'a_pitcher_form_score': -1,
    'h_pitcher_quality_idx': 1, 'a_pitcher_quality_idx': -1,
    # Higher home OPS → better for home
    'h_ops': 1, 'a_ops': -1,
    'h_avg': 1, 'a_avg': -1,
    # Differentials: (home - away), so positive = better for home
    'era_diff': -1,   # higher era_diff = home ERA higher = BAD
    'whip_diff': -1,
    'ops_diff': 1,
    'k_rate_diff': 1,
    'lineup_power_diff': 1,
    'domination_diff': 1,
    # Savant: lower xwoba allowed = better pitcher
    'h_starter_xwoba': -1, 'a_starter_xwoba': 1,
    'h_starter_barrel': -1, 'a_starter_barrel': 1,
    # Higher lineup xwoba = better offense
    'h_lineup_xwoba': 1, 'a_lineup_xwoba': -1,
    'matchup_advantage': 1,
    'xwoba_lineup_diff': 1,
    # Context: unconstrained
    'park_factor': 0, 'is_night': 0,
    'h_rest_days': 0, 'a_rest_days': 0,
    'divisional_game': 0,
    # Momentum
    'elo_diff': 1,
    'streak_diff': 1,
    'h_pyth_pct': 1, 'a_pyth_pct': -1,
    'h_home_wpct': 1, 'a_away_wpct': -1,
    'run_diff_diff': 1,
    'h2h_advantage': 1,
    # Bullpen: higher avail = better
    'h_bp_avail': 1, 'a_bp_avail': -1,
    # Higher bullpen ERA = worse
    'h_bullpen_era_l3': -1, 'a_bullpen_era_l3': 1,
    # MC sim
    'mc_home_prob': 1,
    'mc_margin_expected': 1,
}


def get_monotonic_constraints(feature_list):
    """Build the monotonic constraints tuple for XGBoost."""
    return tuple(MONOTONIC_MAP.get(f, 0) for f in feature_list)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING + FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """Load games from DB, run feature engineering, return df with features."""
    print("═" * 70)
    print(" 📊 LOADING DATA FROM DATABASE")
    print("═" * 70)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT * FROM historico_partidos
        WHERE h_runs_total IS NOT NULL AND a_runs_total IS NOT NULL
        ORDER BY date
    """, conn)
    pp = pd.read_sql("SELECT * FROM pitcher_performances ORDER BY date", conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'], format='mixed')
    pp['date'] = pd.to_datetime(pp['date'], format='mixed')

    print(f"  Games loaded: {len(df)}")
    print(f"  Pitcher records: {len(pp)}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Run feature engineering
    print("\n  ⚙️  Building features...")
    t0 = time.time()
    df, all_feats, target = build_all_features(df, pp, predict_mode=False)
    elapsed = time.time() - t0
    print(f"  ✅ Features built in {elapsed:.1f}s ({len(all_feats)} total features)")
    print(f"  📏 Dataset after feature engineering: {len(df)} games")

    return df, target


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
def prepare_features(df, target, feature_list):
    """
    Temporal split + median imputation.
    Returns X_train, y_train, X_val, y_val, X_test, y_test, medians dict.
    """
    print("\n" + "═" * 70)
    print(" 🔪 TEMPORAL SPLIT + FEATURE PREP")
    print("═" * 70)

    # Check which features exist
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features (will be excluded): {missing}")
    feature_list = available
    print(f"  Using {len(feature_list)} features")

    # ── Temporal split ──
    train_mask = df['date'] < '2025-01-01'     # 2023-2024
    val_mask = (df['date'] >= '2025-01-01') & (df['date'] < '2026-01-01')  # 2025
    test_mask = df['date'] >= '2026-01-01'      # 2026

    X_train_raw = df.loc[train_mask, feature_list].copy()
    X_val_raw = df.loc[val_mask, feature_list].copy()
    X_test_raw = df.loc[test_mask, feature_list].copy()

    y_train = target[train_mask.values]
    y_val = target[val_mask.values]
    y_test = target[test_mask.values]

    print(f"  Train: {len(X_train_raw)} games (2023-2024)")
    print(f"  Val:   {len(X_val_raw)} games (2025)")
    print(f"  Test:  {len(X_test_raw)} games (2026)")
    print(f"  Home win rate — Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")

    # ── Median imputation (computed on TRAIN only, applied to all) ──
    medians = {}
    for col in feature_list:
        med = X_train_raw[col].median()
        if pd.isna(med):
            med = 0.0  # Last resort fallback
        medians[col] = float(med)

    X_train = X_train_raw.fillna(medians).values.astype(np.float32)
    X_val = X_val_raw.fillna(medians).values.astype(np.float32)
    X_test = X_test_raw.fillna(medians).values.astype(np.float32)

    # Sanity check: no NaN/Inf
    for name, X in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️  {name} has {nan_count} NaN, {inf_count} Inf — replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  ✅ Imputation complete (medians computed on train set only)")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_list, medians


# ══════════════════════════════════════════════════════════════════════════════
# 5. OPTUNA HYPERPARAMETER OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
def run_optuna(X_train, y_train, X_val, y_val, feature_list, n_trials=100):
    """Run Optuna HPO, return best params."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "═" * 70)
    print(f" 🔬 OPTUNA HYPERPARAMETER SEARCH ({n_trials} trials)")
    print("═" * 70)

    mc = get_monotonic_constraints(feature_list)
    best_score = [float('inf')]
    best_trial_num = [0]

    def objective(trial):
        params = {
            'n_estimators': 1500,  # Large, will early-stop
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'monotone_constraints': mc,
            'early_stopping_rounds': 50,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
        }

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_proba = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, y_proba)

        if ll < best_score[0]:
            best_score[0] = ll
            best_trial_num[0] = trial.number
            acc = accuracy_score(y_val, (y_proba >= 0.5).astype(int))
            print(f"  🏆 Trial {trial.number:3d}: log_loss={ll:.5f}, acc={acc:.4f} "
                  f"(depth={params['max_depth']}, lr={params['learning_rate']:.4f})")

        return ll

    study = optuna.create_study(direction='minimize', study_name='xgb_v2_mlb')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\n  ✅ Best trial: #{study.best_trial.number}")
    print(f"     Log loss: {study.best_value:.5f}")
    print(f"     Params: {json.dumps(study.best_params, indent=2)}")

    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN FINAL MODEL
# ══════════════════════════════════════════════════════════════════════════════
def train_final_model(X_train, y_train, X_val, y_val, feature_list, best_params):
    """Train the final model with best params + early stopping."""
    print("\n" + "═" * 70)
    print(" 🏗️  TRAINING FINAL MODEL")
    print("═" * 70)

    mc = get_monotonic_constraints(feature_list)

    final_params = {
        'n_estimators': 1500,
        'early_stopping_rounds': 50,
        'monotone_constraints': mc,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        **best_params,
    }

    model = XGBClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    n_trees = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else model.n_estimators
    print(f"  Trees used: {n_trees}")
    print(f"  Parameters: {json.dumps(best_params, indent=2)}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 7. CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_model(model, X_val, y_val):
    """Isotonic calibration on the validation set using manual IsotonicRegression."""
    from sklearn.isotonic import IsotonicRegression
    print("\n  🎯 Calibrating probabilities (isotonic regression on val set)...")
    
    # Get raw probabilities from the model
    y_proba_raw = model.predict_proba(X_val)[:, 1]
    
    # Fit isotonic regression: maps raw probabilities → calibrated probabilities
    iso_reg = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
    iso_reg.fit(y_proba_raw, y_val)
    
    # Verify calibration improved
    from sklearn.metrics import log_loss as ll_fn
    y_cal = iso_reg.predict(y_proba_raw)
    ll_before = ll_fn(y_val, y_proba_raw)
    ll_after = ll_fn(y_val, y_cal)
    print(f"    Log loss: {ll_before:.5f} (raw) → {ll_after:.5f} (calibrated)")
    
    return iso_reg



# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, X, y, label, iso_calibrator=None):
    """Full evaluation: accuracy, log_loss, Brier, calibration buckets."""
    y_pred = model.predict(X)
    y_proba_raw = model.predict_proba(X)[:, 1]

    if iso_calibrator is not None:
        y_proba_cal = iso_calibrator.predict(y_proba_raw)
    else:
        y_proba_cal = y_proba_raw


    acc = accuracy_score(y, y_pred)
    ll_raw = log_loss(y, y_proba_raw)
    ll_cal = log_loss(y, y_proba_cal) if iso_calibrator is not None else ll_raw
    brier = brier_score_loss(y, y_proba_cal)

    print(f"\n  ── {label} ──")
    print(f"  Accuracy:       {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Log Loss (raw): {ll_raw:.5f}")
    if iso_calibrator is not None:
        print(f"  Log Loss (cal): {ll_cal:.5f}")
    print(f"  Brier Score:    {brier:.5f}")

    # Calibration buckets
    print(f"\n  Calibration Curve ({label}):")
    print(f"  {'Bucket':<12} {'Predicted':>10} {'Actual':>10} {'Count':>8} {'Gap':>8}")
    print(f"  {'─'*50}")
    confs = np.maximum(y_proba_cal, 1 - y_proba_cal) * 100
    picks = (y_proba_cal >= 0.5).astype(int)
    hits = (picks == y).astype(int)

    for lo in range(50, 85, 5):
        hi = lo + 5
        mask = (confs >= lo) & (confs < hi)
        n = mask.sum()
        if n >= 5:
            actual_pct = hits[mask].mean() * 100
            predicted_pct = (lo + hi) / 2
            gap = actual_pct - predicted_pct
            emoji = "✅" if abs(gap) < 5 else "⚠️"
            print(f"  {lo}-{hi}%{'':<7} {predicted_pct:>8.1f}%  {actual_pct:>8.1f}%  {n:>6}  {gap:>+6.1f}% {emoji}")

    # 75%+ bucket
    mask_high = confs >= 75
    n_high = mask_high.sum()
    if n_high >= 3:
        actual_high = hits[mask_high].mean() * 100
        print(f"  75%+{'':<7} {'>75':>9}%  {actual_high:>8.1f}%  {n_high:>6}")

    return {
        'accuracy': acc,
        'log_loss_raw': ll_raw,
        'log_loss_calibrated': ll_cal,
        'brier': brier,
    }


def print_feature_importance(model, feature_list, top_n=25):
    """Print top N features by importance."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  ── Feature Importance (Top {top_n}) ──")
    print(f"  {'Rank':<6} {'Feature':<35} {'Importance':>12} {'Monotonic':>10}")
    print(f"  {'─'*65}")
    for i, idx in enumerate(sorted_idx[:top_n]):
        fname = feature_list[idx]
        imp = importances[idx]
        mc_val = MONOTONIC_MAP.get(fname, 0)
        mc_str = {1: '↑ (+1)', -1: '↓ (-1)', 0: '∅ (0)'}[mc_val]
        bar = '█' * int(imp * 200)
        print(f"  {i+1:<6} {fname:<35} {imp:>10.4f}   {mc_str:<10} {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def save_artifacts(model, iso_calibrator, feature_list, medians, best_params, metrics):
    """Save everything needed for inference."""
    print("\n" + "═" * 70)
    print(" 💾 SAVING MODEL ARTIFACTS")
    print("═" * 70)

    # Model
    joblib.dump(model, os.path.join(V2_DIR, 'xgb_v2.pkl'))
    print(f"  ✅ Raw model → {V2_DIR}/xgb_v2.pkl")

    # Isotonic calibrator
    joblib.dump(iso_calibrator, os.path.join(V2_DIR, 'xgb_v2_calibrator.pkl'))
    print(f"  ✅ Isotonic calibrator → {V2_DIR}/xgb_v2_calibrator.pkl")


    # Feature list
    with open(os.path.join(V2_DIR, 'features_v2.json'), 'w') as f:
        json.dump(feature_list, f, indent=2)
    print(f"  ✅ Feature list ({len(feature_list)} features) → features_v2.json")

    # Medians (for imputation at inference)
    with open(os.path.join(V2_DIR, 'medians_v2.json'), 'w') as f:
        json.dump(medians, f, indent=2)
    print(f"  ✅ Imputation medians → medians_v2.json")

    # Hyperparameters
    with open(os.path.join(V2_DIR, 'best_params_v2.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"  ✅ Best hyperparameters → best_params_v2.json")

    # Metrics
    with open(os.path.join(V2_DIR, 'metrics_v2.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ Evaluation metrics → metrics_v2.json")

    # Monotonic constraints used
    mc = {f: MONOTONIC_MAP.get(f, 0) for f in feature_list}
    with open(os.path.join(V2_DIR, 'monotonic_v2.json'), 'w') as f:
        json.dump(mc, f, indent=2)
    print(f"  ✅ Monotonic constraints → monotonic_v2.json")

    # Train date
    with open(os.path.join(V2_DIR, 'train_date.txt'), 'w') as f:
        f.write(datetime.now().isoformat())

    print(f"\n  📁 All artifacts saved to: {V2_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(n_trials=100):
    print("╔" + "═" * 68 + "╗")
    print("║  XGBoost V2 — Proper Single Model for MLB Game Prediction         ║")
    print("╚" + "═" * 68 + "╝")
    t_start = time.time()

    # 1. Load data
    df, target = load_data()

    # 2. Prepare features (temporal split + median imputation)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_list, medians = \
        prepare_features(df, target, FEATURES_V2)

    # 3. Optuna HPO
    best_params = run_optuna(X_train, y_train, X_val, y_val, feature_list, n_trials=n_trials)

    # 4. Train final model with best params
    model = train_final_model(X_train, y_train, X_val, y_val, feature_list, best_params)

    # 5. Calibrate
    calibrated = calibrate_model(model, X_val, y_val)

    # 6. Evaluate on all splits
    print("\n" + "═" * 70)
    print(" 📈 EVALUATION RESULTS")
    print("═" * 70)

    metrics = {}
    metrics['train'] = evaluate(model, X_train, y_train, 'TRAIN', calibrated)
    metrics['val'] = evaluate(model, X_val, y_val, 'VALIDATION (2025)', calibrated)
    metrics['test'] = evaluate(model, X_test, y_test, 'TEST (2026) — UNSEEN', calibrated)

    # Overfit check
    overfit_gap = metrics['train']['accuracy'] - metrics['val']['accuracy']
    print(f"\n  🔍 Overfit gap (Train - Val): {overfit_gap*100:.1f}%", end="")
    if overfit_gap < 0.03:
        print(" ✅ Healthy")
    elif overfit_gap < 0.05:
        print(" ⚠️ Mild overfitting")
    else:
        print(" 🚨 Significant overfitting!")

    # 7. Feature importance
    print_feature_importance(model, feature_list)

    # 8. Save
    save_artifacts(model, calibrated, feature_list, medians, best_params, metrics)

    elapsed = time.time() - t_start
    print(f"\n  ⏱️  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\n" + "═" * 70)
    print("  ✅ DONE — Model V2 trained, calibrated, and saved.")
    print("═" * 70)

    return model, calibrated, feature_list, metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train XGBoost V2 model')
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--quick', action='store_true', help='Quick run with 20 trials')
    args = parser.parse_args()

    n = 20 if args.quick else args.trials
    main(n_trials=n)
