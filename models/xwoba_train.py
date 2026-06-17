"""
omega_xwoba_train.py — ML xwOBA Predictor Training (REVERSIBLE / TEST ONLY)
============================================================================
Trains XGBoost model to predict single-game batter xwOBA using:
  - Batter rolling savant stats (xwOBA, barrel%, hardhit%, EV, bat_speed, etc.)
  - Pitcher rolling savant stats (xwOBA allowed, barrel% allowed, EV allowed, spin)
  - Platoon split (batter L/R vs pitcher L/R)
  - Park factor
  - Player ELO ratings
  - Head-to-head history

Usage:
  python3 omega_xwoba_train.py --mode train
  python3 omega_xwoba_train.py --mode backtest
  python3 omega_xwoba_train.py --mode predict --date 2026-06-03
"""

import sqlite3, argparse, warnings, json, pickle, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

DB_PATH = CONFIG.paths.db
MODEL_DIR = CONFIG.paths.xwoba
MODEL_DIR.mkdir(exist_ok=True)

MLB_AVG_XWOBA = 0.310

PARK_FACTORS = {
    'Colorado Rockies': 1.15, 'Cincinnati Reds': 1.06, 'Texas Rangers': 1.04,
    'Baltimore Orioles': 1.03, 'Boston Red Sox': 1.03, 'New York Yankees': 1.02,
    'Philadelphia Phillies': 1.01, 'Chicago Cubs': 1.00, 'Atlanta Braves': 0.99,
    'Minnesota Twins': 0.99, 'Milwaukee Brewers': 0.98, 'St. Louis Cardinals': 0.98,
    'Los Angeles Dodgers': 0.97, 'Pittsburgh Pirates': 0.97, 'Tampa Bay Rays': 0.97,
    'Toronto Blue Jays': 0.97, 'San Francisco Giants': 0.96, 'New York Mets': 0.96,
    'Arizona Diamondbacks': 0.96, 'Cleveland Guardians': 0.96, 'Houston Astros': 0.96,
    'Chicago White Sox': 0.95, 'Kansas City Royals': 0.95, 'Detroit Tigers': 0.95,
    'Los Angeles Angels': 0.95, 'Oakland Athletics': 0.95, 'Miami Marlins': 0.94,
    'San Diego Padres': 0.93, 'Seattle Mariners': 0.91, 'Washington Nationals': 0.95,
}


def rolling_stats(conn, player_id, before_date, table='batter', span=10):
    """Compute rolling stats for a player from savant data."""
    if table == 'batter':
        rows = conn.execute(f'''
            SELECT game_date, avg_xwoba, barrels, hard_hits, avg_ev, bbe,
                   avg_bat_speed, avg_swing_length, sweet_spot_count,
                   z_swings, z_pitches, o_swings, o_pitches, p_throws
            FROM savant_batter_daily
            WHERE player_id = ? AND game_date < ? AND bbe >= 2
            ORDER BY game_date DESC LIMIT ?
        ''', (int(player_id), before_date, span)).fetchall()
        if not rows:
            return None
        # Exponential decay
        stats = {}
        for key_idx, key in enumerate(['xwoba','barrel','hardhit','ev','bbe',
                                        'bat_speed','swing_length','sweet_spot',
                                        'z_swings','z_pitches','o_swings','o_pitches']):
            vals = [r[key_idx+1] for r in rows if r[key_idx+1] is not None]
            if vals:
                weights = [0.7**i for i in range(len(vals))]
                stats[key] = np.average(vals, weights=weights[:len(vals)])
            else:
                stats[key] = 0
        # Derived rates
        bbe = max(stats['bbe'], 1)
        stats['barrel_rate'] = stats['barrel'] / bbe
        stats['hardhit_rate'] = stats['hardhit'] / bbe
        stats['sweet_spot_rate'] = stats['sweet_spot'] / bbe if stats['sweet_spot'] > 0 else 0
        stats['z_swing_rate'] = stats['z_swings'] / max(stats['z_pitches'], 1) if stats['z_pitches'] > 0 else 0
        stats['o_swing_rate'] = stats['o_swings'] / max(stats['o_pitches'], 1) if stats['o_pitches'] > 0 else 0
        stats['games'] = len(rows)
        stats['p_throws'] = rows[0][13] if rows[0][13] else 'R'
        return stats

    elif table == 'pitcher':
        rows = conn.execute(f'''
            SELECT game_date, avg_xwoba_allowed, barrels_allowed, hard_hits_allowed,
                   avg_ev_allowed, bbe, avg_release_spin_rate, avg_release_extension, stand
            FROM savant_pitcher_daily
            WHERE player_id = ? AND game_date < ? AND bbe >= 2
            ORDER BY game_date DESC LIMIT ?
        ''', (int(player_id), before_date, span)).fetchall()
        if not rows:
            return None
        stats = {}
        for key_idx, key in enumerate(['xwoba_allowed','barrels_allowed','hardhit_allowed',
                                        'ev_allowed','bbe','spin_rate','extension']):
            vals = [r[key_idx+1] for r in rows if r[key_idx+1] is not None]
            if vals:
                weights = [0.7**i for i in range(len(vals))]
                stats[key] = np.average(vals, weights=weights[:len(vals)])
            else:
                stats[key] = 0
        bbe = max(stats['bbe'], 1)
        stats['barrel_allowed_rate'] = stats['barrels_allowed'] / bbe
        stats['hardhit_allowed_rate'] = stats['hardhit_allowed'] / bbe
        stats['games'] = len(rows)
        stats['stand'] = rows[0][8] if rows[0][8] else 'R'
        return stats

    return None


def build_dataset(conn, start_date='2023-04-01', end_date='2026-06-02'):
    """
    Build training dataset: for each (batter, date, pitcher) triple,
    compute features and target (actual xwOBA).
    """
    print("Building dataset...")

    # Get all batter games with actual xwOBA
    batter_games = conn.execute('''
        SELECT sbd.player_id, sbd.game_date, sbd.avg_xwoba, sbd.p_throws, sbd.bbe,
               bp.game_pk, bp.team_id
        FROM savant_batter_daily sbd
        JOIN batter_performances bp ON sbd.player_id = bp.player_id AND sbd.game_date = bp.date
        WHERE sbd.game_date >= ? AND sbd.game_date <= ? AND sbd.bbe >= 3 AND sbd.avg_xwoba IS NOT NULL
    ''', (start_date, end_date)).fetchall()

    print(f"Batter games with savant data: {len(batter_games)}")

    # For each batter game, find the opposing pitcher (starter from same game_pk)
    dataset = []
    count = 0
    for b_pid, b_date, b_xwoba, b_throws, b_bbe, gpk, b_tid in batter_games:
        # Find opposing starter
        opp = conn.execute('''
            SELECT pp.player_id, pp.team_id
            FROM pitcher_performances pp
            WHERE pp.game_pk = ? AND pp.role = 'starter' AND pp.team_id != ?
            LIMIT 1
        ''', (gpk, b_tid)).fetchone()

        if not opp:
            continue

        p_pid = opp[0]

        # Get game info for park
        game = conn.execute('''
            SELECT home_team, home_team_id FROM historico_partidos WHERE game_pk = ?
        ''', (gpk,)).fetchone()
        if not game:
            continue
        home_team, home_tid = game
        is_home = (b_tid == home_tid)

        # Get pitcher rolling stats
        p_stats = rolling_stats(conn, p_pid, b_date, table='pitcher', span=10)
        if not p_stats:
            continue

        # Get batter rolling stats (exclude current game)
        b_stats = rolling_stats(conn, b_pid, b_date, table='batter', span=10)
        if not b_stats:
            continue

        # Park factor
        park = PARK_FACTORS.get(home_team, 1.00)

        # Platoon: batter vs pitcher
        # b_throws = batter's handedness (L/R), p_stats['stand'] = pitcher throws L/R
        same_hand = 1 if b_throws == p_stats['stand'] else 0

        # ELO ratings
        b_elo = conn.execute('''
            SELECT elo FROM player_elo WHERE player_id = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        ''', (b_pid, b_date)).fetchone()
        p_elo = conn.execute('''
            SELECT elo FROM player_elo WHERE player_id = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        ''', (p_pid, b_date)).fetchone()

        b_elo_val = b_elo[0] if b_elo else 1500
        p_elo_val = p_elo[0] if p_elo else 1500

        # Assemble features
        features = {
            # Batter rolling
            'b_xwoba': b_stats['xwoba'],
            'b_barrel_rate': b_stats['barrel_rate'],
            'b_hardhit_rate': b_stats['hardhit_rate'],
            'b_ev': b_stats['ev'],
            'b_bat_speed': b_stats['bat_speed'],
            'b_swing_length': b_stats['swing_length'],
            'b_sweet_spot_rate': b_stats['sweet_spot_rate'],
            'b_z_swing_rate': b_stats['z_swing_rate'],
            'b_o_swing_rate': b_stats['o_swing_rate'],
            'b_games': b_stats['games'],

            # Pitcher rolling
            'p_xwoba_allowed': p_stats['xwoba_allowed'],
            'p_barrel_allowed_rate': p_stats['barrel_allowed_rate'],
            'p_hardhit_allowed_rate': p_stats['hardhit_allowed_rate'],
            'p_ev_allowed': p_stats['ev_allowed'],
            'p_spin_rate': p_stats['spin_rate'],
            'p_extension': p_stats['extension'],
            'p_games': p_stats['games'],

            # Matchup
            'same_hand': same_hand,
            'park_factor': park,
            'is_home': 1 if is_home else 0,

            # ELO
            'b_elo': b_elo_val,
            'p_elo': p_elo_val,
            'elo_diff': (b_elo_val - p_elo_val) / 100.0,

            # Interactions
            'b_xwoba_x_p_xwoba': b_stats['xwoba'] * p_stats['xwoba_allowed'],
            'b_barrel_x_p_barrel': b_stats['barrel_rate'] * p_stats['barrel_allowed_rate'],
        }

        dataset.append({**features, 'target': b_xwoba, 'batter_id': b_pid, 'date': b_date, 'game_pk': gpk})

        count += 1
        if count % 5000 == 0:
            print(f"  Processed {count} samples...")

    print(f"Total dataset: {len(dataset)} samples")
    return pd.DataFrame(dataset)


def train_model(df):
    """Train XGBoost model on the dataset."""
    feature_cols = [c for c in df.columns if c not in ('target','batter_id','date','game_pk')]
    X = df[feature_cols].values
    y = df['target'].values

    print(f"\nTraining on {len(X)} samples, {len(feature_cols)} features")
    print(f"Target stats: mean={y.mean():.3f}, std={y.std():.3f}, min={y.min():.3f}, max={y.max():.3f}")

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=10,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0,1]

    print(f"\n=== TEST SET RESULTS ===")
    print(f"Samples: {len(y_test)}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Corr: {corr:.4f}")

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n=== TOP 15 FEATURES ===")
    for i in sorted_idx[:15]:
        print(f"  {feature_cols[i]:30s} {importances[i]:.4f}")

    # Save model
    model_path = MODEL_DIR / 'xwoba_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save feature list
    with open(MODEL_DIR / 'features.json', 'w') as f:
        json.dump(feature_cols, f)

    # Save metrics
    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2, 'corr': corr, 'n_train': len(X_train), 'n_test': len(X_test)}
    with open(MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {model_path}")
    return model, feature_cols, metrics


def backtest_model(conn, model, feature_cols, start_date='2026-04-01', end_date='2026-06-02'):
    """Backtest the trained model on recent games."""
    print(f"\nBacktesting from {start_date} to {end_date}...")

    batter_games = conn.execute('''
        SELECT sbd.player_id, sbd.game_date, sbd.avg_xwoba, sbd.p_throws, sbd.bbe,
               bp.game_pk, bp.team_id
        FROM savant_batter_daily sbd
        JOIN batter_performances bp ON sbd.player_id = bp.player_id AND sbd.game_date = bp.date
        WHERE sbd.game_date >= ? AND sbd.game_date <= ? AND sbd.bbe >= 3 AND sbd.avg_xwoba IS NOT NULL
    ''', (start_date, end_date)).fetchall()

    results = []
    for b_pid, b_date, b_xwoba, b_throws, b_bbe, gpk, b_tid in batter_games:
        opp = conn.execute('''
            SELECT pp.player_id, pp.team_id FROM pitcher_performances pp
            WHERE pp.game_pk = ? AND pp.role = 'starter' AND pp.team_id != ? LIMIT 1
        ''', (gpk, b_tid)).fetchone()
        if not opp:
            continue

        p_pid = opp[0]
        game = conn.execute('SELECT home_team, home_team_id FROM historico_partidos WHERE game_pk = ?', (gpk,)).fetchone()
        if not game:
            continue
        home_team, home_tid = game
        is_home = (b_tid == home_tid)

        p_stats = rolling_stats(conn, p_pid, b_date, table='pitcher', span=10)
        b_stats = rolling_stats(conn, b_pid, b_date, table='batter', span=10)
        if not p_stats or not b_stats:
            continue

        park = PARK_FACTORS.get(home_team, 1.00)
        same_hand = 1 if b_throws == p_stats['stand'] else 0

        b_elo = conn.execute('SELECT elo FROM player_elo WHERE player_id = ? AND date <= ? ORDER BY date DESC LIMIT 1', (b_pid, b_date)).fetchone()
        p_elo = conn.execute('SELECT elo FROM player_elo WHERE player_id = ? AND date <= ? ORDER BY date DESC LIMIT 1', (p_pid, b_date)).fetchone()
        b_elo_val = b_elo[0] if b_elo else 1500
        p_elo_val = p_elo[0] if p_elo else 1500

        features = {
            'b_xwoba': b_stats['xwoba'], 'b_barrel_rate': b_stats['barrel_rate'],
            'b_hardhit_rate': b_stats['hardhit_rate'], 'b_ev': b_stats['ev'],
            'b_bat_speed': b_stats['bat_speed'], 'b_swing_length': b_stats['swing_length'],
            'b_sweet_spot_rate': b_stats['sweet_spot_rate'],
            'b_z_swing_rate': b_stats['z_swing_rate'], 'b_o_swing_rate': b_stats['o_swing_rate'],
            'b_games': b_stats['games'],
            'p_xwoba_allowed': p_stats['xwoba_allowed'], 'p_barrel_allowed_rate': p_stats['barrel_allowed_rate'],
            'p_hardhit_allowed_rate': p_stats['hardhit_allowed_rate'], 'p_ev_allowed': p_stats['ev_allowed'],
            'p_spin_rate': p_stats['spin_rate'], 'p_extension': p_stats['extension'],
            'p_games': p_stats['games'], 'same_hand': same_hand, 'park_factor': park,
            'is_home': 1 if is_home else 0, 'b_elo': b_elo_val, 'p_elo': p_elo_val,
            'elo_diff': (b_elo_val - p_elo_val) / 100.0,
            'b_xwoba_x_p_xwoba': b_stats['xwoba'] * p_stats['xwoba_allowed'],
            'b_barrel_x_p_barrel': b_stats['barrel_rate'] * p_stats['barrel_allowed_rate'],
        }

        x_vec = np.array([[features.get(fc, 0) for fc in feature_cols]])
        pred = model.predict(x_vec)[0]

        results.append({
            'batter_id': b_pid, 'date': b_date, 'game_pk': gpk,
            'actual': b_xwoba, 'predicted': pred,
            'b_rolling_xwoba': b_stats['xwoba'],
            'p_xwoba_allowed': p_stats['xwoba_allowed'],
            'abs_error': abs(b_xwoba - pred)
        })

    if not results:
        print("No results.")
        return

    actuals = np.array([r['actual'] for r in results])
    preds = np.array([r['predicted'] for r in results])

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    corr = np.corrcoef(actuals, preds)[0,1]
    r2 = r2_score(actuals, preds)
    bias = np.mean(preds - actuals)

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS ({start_date} to {end_date})")
    print(f"{'='*60}")
    print(f"Samples: {len(results)}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Corr: {corr:.4f}")
    print(f"Bias: {bias:+.4f}")

    # Bucket analysis
    print(f"\n=== ACCURACY BY PREDICTED xwOBA BUCKET ===")
    buckets = [(0.150, 0.250), (0.250, 0.300), (0.300, 0.350), (0.350, 0.400), (0.400, 0.600)]
    for lo, hi in buckets:
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() > 0:
            b_mae = mean_absolute_error(actuals[mask], preds[mask])
            b_corr = np.corrcoef(actuals[mask], preds[mask])[0,1] if mask.sum() > 3 else 0
            b_bias = np.mean(preds[mask] - actuals[mask])
            print(f"  {lo:.3f}-{hi:.3f}: n={mask.sum():5d}, MAE={b_mae:.4f}, Corr={b_corr:.3f}, Bias={b_bias:+.4f}")

    # Comparison: just predict batter's rolling average
    b_rolling = np.array([r['b_rolling_xwoba'] for r in results])
    mae_baseline = mean_absolute_error(actuals, b_rolling)
    corr_baseline = np.corrcoef(actuals, b_rolling)[0,1]
    print(f"\n=== BASELINE (batter rolling avg only) ===")
    print(f"MAE:  {mae_baseline:.4f}")
    print(f"Corr: {corr_baseline:.4f}")
    print(f"\nML vs Baseline improvement: MAE {(mae_baseline-mae)/mae_baseline*100:+.1f}%, Corr {corr-corr_baseline:+.4f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xwOBA ML Predictor')
    parser.add_argument('--mode', choices=['train', 'backtest', 'predict'], default='train')
    parser.add_argument('--date', type=str, default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH))

    if args.mode == 'train':
        df = build_dataset(conn)
        model, feature_cols, metrics = train_model(df)
        backtest_model(conn, model, feature_cols)

    elif args.mode == 'backtest':
        model_path = MODEL_DIR / 'xwoba_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(MODEL_DIR / 'features.json') as f:
            feature_cols = json.load(f)
        backtest_model(conn, model, feature_cols)

    elif args.mode == 'predict':
        # Load model
        model_path = MODEL_DIR / 'xwoba_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(MODEL_DIR / 'features.json') as f:
            feature_cols = json.load(f)

        from models.xwoba import predict_today, print_predictions
        results = predict_today(args.date)
        # TODO: enhance with ML predictions
        if results:
            print_predictions(results)

    conn.close()
