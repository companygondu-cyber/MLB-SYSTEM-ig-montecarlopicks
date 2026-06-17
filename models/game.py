import os, json, joblib, warnings, logging, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from datetime import datetime
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

def get_models_dir(prefix=""):
    if prefix:
        d = str(CONFIG.paths.trained / f'{prefix}')
    else:
        d = str(CONFIG.paths.ensemble)
    os.makedirs(d, exist_ok=True)
    return d

MODELS_DIR = get_models_dir()

# Config C Optimal: 6 Elite models (only rf1 amputated)
# 74.21% acc | +2,149u profit | 9/9 calibration ✅
WEIGHTS = {'xgb1': 0.24, 'xgb2': 0.20, 'hgb1': 0.18, 'hgb2': 0.12, 'rf2': 0.06, 'mlp': 0.20}

WINDOWS = {'all': None, 'w50': 750, 'w25': 375, 'w10': 150}

def _make_models():
    return {
        'xgb1': XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                    eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1),
        'xgb2': XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                    subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0,
                    eval_metric='logloss', verbosity=0, random_state=123, n_jobs=-1),
        'hgb1': HistGradientBoostingClassifier(max_iter=600, max_depth=6, learning_rate=0.04,
                    l2_regularization=1.5, min_samples_leaf=15, random_state=42),
        'hgb2': HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.03,
                    l2_regularization=3.0, min_samples_leaf=20, random_state=123),
        'rf2':  RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=12,
                    class_weight='balanced', random_state=123, n_jobs=-1),
        'mlp':  MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=800, random_state=42,
                    early_stopping=True, validation_fraction=0.15, alpha=0.001),

    }

def train_multiwindow(df, features, target, save=True, prefix=""):
    """Train primary ensemble (all data) + window specialists for consensus tag."""
    global WEIGHTS
    weights = WEIGHTS
    print(f"Training {len(features)} features, {len(df)} games... (weights: {prefix or 'MLB'})")

    # Build calibration from temporal split: pre-2026 train, 2026 calibrate
    cal_mask = df['date'] >= '2026-01-01'
    train_mask = df['date'] < '2026-01-01'
    if cal_mask.sum() >= 50:
        X_tr = df[train_mask][features].fillna(0).values
        X_te = df[cal_mask][features].fillna(0).values
        y_tr, y_te = target[train_mask.values], target[cal_mask.values]
        cal_year = '2026'
    else:
        split = int(len(df) * 0.8)
        X_tr = df.iloc[:split][features].fillna(0).values
        X_te = df.iloc[split:][features].fillna(0).values
        y_tr, y_te = target[:split], target[split:]
        cal_year = 'last20pct'
    sc_cal = StandardScaler(); X_tr_s = sc_cal.fit_transform(X_tr); X_te_s = sc_cal.transform(X_te)

    cal_preds, cal_actuals = [], []
    meta_X = []
    meta_models = _make_models()
    for mname, model in meta_models.items():
        model.fit(X_tr_s, y_tr)
        preds = model.predict_proba(X_te_s)[:,1]
        cal_preds.extend(preds.tolist())
        cal_actuals.extend(y_te.tolist())
        meta_X.append(preds)
        
    # Train Stacking Meta-Learner (Logistic Regression on OOF predictions)
    meta_X_arr = np.column_stack(meta_X)
    meta_learner = LogisticRegression(C=0.1)
    meta_learner.fit(meta_X_arr, y_te)
    
    # Extract learned weights
    learned_weights = meta_learner.coef_[0]
    # Convert to softmax-like distribution for interpretability
    exp_w = np.exp(learned_weights)
    dist_w = exp_w / np.sum(exp_w)
    WEIGHTS = {mname: float(dist_w[i]) for i, mname in enumerate(meta_models.keys())}
    print(f"  🧠 Stacking weights learned: {WEIGHTS}")

    # ── META-RF CALIBRATOR ──
    # Predicts confidence based on model disagreements and raw probabilities
    meta_rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
    # Target: did the stacked prediction hit?
    stacked_preds = meta_learner.predict_proba(meta_X_arr)[:,1]
    stacked_picks = (stacked_preds >= 0.5).astype(int)
    meta_rf_target = (stacked_picks == y_te).astype(int)
    meta_rf.fit(meta_X_arr, meta_rf_target)

    val_probs = np.zeros(len(y_te))
    for i, mname in enumerate(meta_models.keys()):
        val_probs += meta_X_arr[:, i] * WEIGHTS[mname]

    confs_cal = np.maximum(val_probs, 1 - val_probs) * 100
    picks_cal = (val_probs >= 0.5).astype(int)
    hits_cal = (picks_cal == y_te).astype(int)

    calibration = {}
    for lo in range(50, 100, 5):
        hi = lo + 5
        mask = (confs_cal >= lo) & (confs_cal < hi)
        if mask.sum() >= 10:
            calibration[f'{lo}_{hi}'] = round(float(hits_cal[mask].mean()) * 100, 1)

    print(f"Bucket Calibration on {cal_year} ({X_te.shape[0]} games): {calibration}")

    if save:
        # Train ALL windows on full data
        for wname, wsize in WINDOWS.items():
            if wsize and len(df) < wsize:
                logging.warning(f"Window size {wsize} exceeds dataframe length {len(df)} for window '{wname}'.")
            dw = df.iloc[-wsize:] if wsize and len(df) > wsize else df
            tw = target[-wsize:] if wsize and len(df) > wsize else target
            X = dw[features].fillna(0).values
            scaler = StandardScaler(); X_s = scaler.fit_transform(X)
            for mname, model in _make_models().items():
                model.fit(X_s, tw)
                joblib.dump(model, os.path.join(get_models_dir(prefix), f'{wname}_{mname}.pkl'))
            joblib.dump(scaler, os.path.join(get_models_dir(prefix), f'scaler_{wname}.pkl'))
            if wname == 'all':
                joblib.dump(meta_learner, os.path.join(get_models_dir(prefix), f'meta_learner.pkl'))
                joblib.dump(meta_rf, os.path.join(get_models_dir(prefix), f'meta_rf.pkl'))
                with open(os.path.join(get_models_dir(prefix),'weights.json'),'w') as f: json.dump(WEIGHTS,f)

        with open(os.path.join(get_models_dir(prefix),'features.json'),'w') as f: json.dump(features,f)
        with open(os.path.join(get_models_dir(prefix),'calibration.json'),'w') as f: json.dump(calibration,f)
        with open(os.path.join(get_models_dir(prefix),'train_date.txt'),'w') as f: f.write(datetime.now().isoformat())
        print(f"Saved 20 models. Calibration: {calibration}")

    # Return models for backtest use (primary = all window)
    models, scalers = {}, {}
    X_full = df[features].fillna(0).values
    sc_full = StandardScaler(); X_s = sc_full.fit_transform(X_full)
    for mname, model in _make_models().items():
        model.fit(X_s, target)
        models[f'all_{mname}'] = model
    scalers['all'] = sc_full
    # Window specialists
    for wname, wsize in list(WINDOWS.items()):
        if wname == 'all': continue
        if wsize and len(df) > wsize:
            dw = df.iloc[-wsize:]; tw = target[-wsize:]
        else: continue
        sc_w = StandardScaler(); X_w = sc_w.fit_transform(dw[features].fillna(0).values)
        scalers[wname] = sc_w
        for mname, model in _make_models().items():
            model.fit(X_w, tw)
            models[f'{wname}_{mname}'] = model

    return models, scalers, calibration


def load_all_models(prefix=""):
    global WEIGHTS
    with open(os.path.join(get_models_dir(prefix),'features.json')) as f: features = json.load(f)
    cal = {}
    cp = os.path.join(get_models_dir(prefix),'calibration.json')
    if os.path.exists(cp):
        with open(cp) as f: cal = json.load(f)
    models, scalers = {}, {}
    wp = os.path.join(get_models_dir(prefix), 'weights.json')
    if os.path.exists(wp):
        with open(wp) as f: 
            WEIGHTS = json.load(f)
    else:
        WEIGHTS = {'xgb1': 0.22, 'xgb2': 0.16, 'hgb1': 0.16, 'hgb2': 0.10, 'rf2': 0.04, 'mlp': 0.16, }

    for wname in WINDOWS:
        sp = os.path.join(get_models_dir(prefix), f'scaler_{wname}.pkl')
        if not os.path.exists(sp): continue
        scalers[wname] = joblib.load(sp)
        for mname in WEIGHTS:
            mp = os.path.join(get_models_dir(prefix), f'{wname}_{mname}.pkl')
            if os.path.exists(mp): models[f'{wname}_{mname}'] = joblib.load(mp)
            
    # Load meta calibrator if exists
    meta_rf_path = os.path.join(get_models_dir(prefix), 'meta_rf.pkl')
    if os.path.exists(meta_rf_path):
        models['meta_rf'] = joblib.load(meta_rf_path)
        
        
    return models, scalers, features, cal


def ensemble_predict(models, scalers, features, X_dict, calibration=None, weights=None):
    if weights is None:
        weights = WEIGHTS
    if not features or not X_dict:
        return {'prob':0.5,'conf':0,'favorite':'','tags':[],'alpha':0.0,'units':0.0,
                'god_mode':False,'consensus':0,'total_windows':0,'window_probs':{}}

    X_df = pd.DataFrame([X_dict])[features].fillna(0)

    # ── PRIMARY PREDICTION from 'all' window ──
    X_s = scalers['all'].transform(X_df.values)
    indiv_probs = []
    prob = 0.0; wt = 0.0
    for mname, weight in weights.items():
        key = f'all_{mname}'
        if key not in models: continue
        p = models[key].predict_proba(X_s)[0, 1]
        prob += p * weight; wt += weight
        indiv_probs.append(p)
    prob = prob / wt if wt > 0 else 0.5
    conf_raw = round(max(prob, 1-prob) * 100, 1)

    # ── GOD MODE: all 5 models agree strongly ──
    god_mode = False
    if len(indiv_probs) >= 5:
        god_mode = all(p > 0.65 for p in indiv_probs) or all(p < 0.35 for p in indiv_probs)

    # ── WINDOW CONSENSUS (tag only, does NOT change prediction) ──
    window_probs = {'all': prob}
    for wname in ['w50','w25','w10']:
        if wname not in scalers: continue
        X_w = scalers[wname].transform(X_df.values)
        wp = 0.0; wwt = 0.0
        for mname, weight in weights.items():
            key = f'{wname}_{mname}'
            if key not in models: continue
            p = models[key].predict_proba(X_w)[0, 1]
            wp += p * weight; wwt += weight
        if wwt > 0: window_probs[wname] = wp / wwt

    # Count how many windows agree on the same side
    home_side = 1 if prob >= 0.5 else 0
    consensus = sum(1 for w,p in window_probs.items() if (1 if p>=0.5 else 0) == home_side)

    # ── CALIBRATION (bucket only) ──
    conf_calibrated = conf_raw
    if calibration:
        bucket = f'{int(conf_raw//5)*5}_{int(conf_raw//5)*5+5}'
        if bucket in calibration:
            conf_calibrated = calibration[bucket]
            
    # Use meta_rf to adjust confidence if available
    if 'meta_rf' in models:
        meta_X = np.array([indiv_probs])
        prob_correct = models['meta_rf'].predict_proba(meta_X)[0, 1]
        # Adjust confidence: if meta_rf thinks we are very right, boost confidence. If very wrong, tank it.
        # prob_correct ranges ~0.4 to ~0.8. Normalize around 0.6.
        boost = (prob_correct - 0.6) * 100 * 0.15  # Max adjustment ~ +/- 3 (was +/-10, overconfident)
        conf_calibrated = min(99.0, max(50.0, conf_calibrated + boost))

    return {
        'prob': prob,
        'conf_raw': conf_raw,
        'conf_calibrated': round(conf_calibrated, 1),
        'god_mode': god_mode,
        'consensus': consensus,
        'total_windows': len(window_probs),
        'window_probs': {w: round(p*100,1) for w,p in window_probs.items()},
    }


def models_exist(prefix=""):
    return os.path.exists(os.path.join(get_models_dir(prefix),'features.json')) and \
           os.path.exists(os.path.join(get_models_dir(prefix),'scaler_all.pkl'))

def needs_retrain(prefix="", min_days=1):
    path = os.path.join(get_models_dir(prefix),'train_date.txt')
    if not os.path.exists(path): return True
    try:
        with open(path) as f:
            ts = f.read().strip().replace('Z', '+00:00')  # Fix 🟡8: handle Z suffix
            last = datetime.fromisoformat(ts.split('+')[0].split('Z')[0])  # Strip timezone
        return (datetime.now() - last).days >= min_days
    except Exception:
        return True
