#!/usr/bin/env python3
"""
backtest_real.py — Walk-forward backtest sin bugs.
- Mismo pipeline de features para train y predict (build_all_features, sin override manual)
- Batch semanal (train cada 7 días, no por juego individual)
- DB schedule (sin API calls) → rápido
- Cutoff único: train y predict usan data < target_date
"""
import os, sys, sqlite3, csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
DB_PATH = os.path.join(BASE_DIR, 'data', 'omega_2026_BETA.db')

from pipeline import load_data, get_pitcher_stats
from core_features import build_all_features
from models.game import train_multiwindow, ensemble_predict, load_all_models, WEIGHTS

def get_db_schedule(date_str):
    """Get games for a specific date from DB."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT game_pk, home_team, away_team, date, "
        "pp_h_starter_name, pp_a_starter_name, "
        "pp_h_starter_id, pp_a_starter_id, "
        "h_starter_era, a_starter_era, h_starter_whip, a_starter_whip, "
        "h_starter_strikeout_rate, a_starter_strikeout_rate, "
        "h_starter_walk_rate, a_starter_walk_rate, "
        "h_runs_total, a_runs_total "
        "FROM historico_partidos WHERE date = ?", (date_str,)
    ).fetchall()
    conn.close()
    return rows

def get_actual_winner(date_str):
    """Get actual home/away winner from DB."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT home_team, away_team, h_runs_total, a_runs_total "
        "FROM historico_partidos WHERE date = ? "
        "AND h_runs_total IS NOT NULL AND a_runs_total IS NOT NULL",
        (date_str,)
    ).fetchall()
    conn.close()
    results = {}
    for r in rows:
        hn, an, hr, ar = r
        winner = hn if hr > ar else an
        results[(hn, an)] = winner
    return results

def run_backtest_real(start_date, end_date, weeks_per_train=1):
    """
    Walk-forward backtest:
    - Train every `weeks_per_train` weeks on data up to (day_before_first_test)
    - Predict all days in the following `weeks_per_train` weeks
    - Uses DB schedule only (no API)
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Pre-load ALL DB data once (for feature building)
    df_all, pp_all = load_data()
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date').reset_index(drop=True)
    
    curr_train_start = start_dt
    all_results = []
    csv_path = os.path.join(BASE_DIR, 'backtest_real_results.csv')
    pd.DataFrame(columns=['date','home','away','pick','prob','conf','tier','consensus','actual_winner','hit']).to_csv(csv_path, index=False)
    
    while curr_train_start <= end_dt:
        train_end = curr_train_start - timedelta(days=1)
        test_end = min(curr_train_start + timedelta(days=weeks_per_train*7 - 1), end_dt)
        
        train_cutoff_str = train_end.strftime('%Y-%m-%d')
        test_start_str = curr_train_start.strftime('%Y-%m-%d')
        test_end_str = test_end.strftime('%Y-%m-%d')
        
        print(f"\n{'='*60}")
        print(f"Train: ≤{train_cutoff_str}  |  Test: {test_start_str} → {test_end_str}")
        print(f"{'='*60}")
        
        # ── TRAIN ──
        train_mask = df_all['date'] <= pd.to_datetime(train_cutoff_str)
        df_train_raw = df_all[train_mask].copy()
        if len(df_train_raw) < 500:
            print(f"  Not enough training data ({len(df_train_raw)}), skipping")
            curr_train_start = test_end + timedelta(days=1)
            continue
        
        pp_train = pp_all[pp_all['date'] <= pd.to_datetime(train_cutoff_str)].copy() if not pp_all.empty else pd.DataFrame()
        
        print(f"  Building features on {len(df_train_raw)} training games...", flush=True)
        df_feat, feats, target = build_all_features(df_train_raw, pp_train)
        
        tgt = target
        print(f"  Training {len(feats)} features, {len(df_feat)} games...", flush=True)
        models, scalers, cal = train_multiwindow(df_feat, feats, tgt, save=False)
        
        # ── PREDICT each day in test window ──
        curr_test = curr_train_start
        while curr_test <= test_end and curr_test <= end_dt:
            date_str = curr_test.strftime('%Y-%m-%d')
            
            # Get actual results
            actual_results = get_actual_winner(date_str)
            if not actual_results:
                curr_test += timedelta(days=1)
                continue
            
            # Get DB schedule
            db_games = get_db_schedule(date_str)
            if not db_games:
                curr_test += timedelta(days=1)
                continue
            
            # Build feature df for this day's games
            today_rows = []
            for g in db_games:
                row = {
                'game_pk': g[0], 'date': pd.to_datetime(date_str),
                'home_team': g[1], 'away_team': g[2],
                'h_starter': str(g[4] or ''), 'a_starter': str(g[5] or ''),
                'pp_h_starter_id': int(g[6] or 0), 'pp_a_starter_id': int(g[7] or 0),
                'h_starter_era': float(g[8] or 4.50), 'a_starter_era': float(g[9] or 4.50),
                'h_starter_whip': float(g[10] or 1.35), 'a_starter_whip': float(g[11] or 1.35),
                'h_starter_strikeout_rate': float(g[12] or 0.220), 'a_starter_strikeout_rate': float(g[13] or 0.220),
                'h_starter_walk_rate': float(g[14] or 0.085), 'a_starter_walk_rate': float(g[15] or 0.085),
                'h_runs_total': np.nan, 'a_runs_total': np.nan,
            }
                today_rows.append(row)
            
            if not today_rows:
                curr_test += timedelta(days=1)
                continue
            
            df_today = pd.DataFrame(today_rows)
            
            # Historical feature data up to day before test
            hist_mask = df_all['date'] < pd.to_datetime(date_str)
            df_hist = df_all[hist_mask].copy()
            pp_hist = pp_all[pp_all['date'] < pd.to_datetime(date_str)].copy() if not pp_all.empty else pd.DataFrame()
            
            df_combined = pd.concat([df_hist, df_today], ignore_index=True)
            
            df_combined_feat, feats2, _ = build_all_features(df_combined, pp_hist, predict_mode=True)
            
            # Extract today's rows
            today_pks = [r['game_pk'] for r in today_rows]
            df_today_feat = df_combined_feat[df_combined_feat['game_pk'].isin(today_pks)]
            
            # Use models from training (no save/load needed)
            models_live, scalers_live, feats_live, cal_live = models, scalers, feats, cal
            
            for _, row in df_today_feat.iterrows():
                gpk = int(row['game_pk'])
                hn, an = row['home_team'], row['away_team']
                
                # Find this game in db_games for actual results
                match = [g for g in db_games if g[0] == gpk]
                if not match:
                    continue
                
                d = row.to_dict()
                for f in feats_live:
                    if f not in d or pd.isna(d.get(f)):
                        d[f] = 0
                
                r = ensemble_predict(models_live, scalers_live, feats_live, d, cal_live)
                
                pick = hn if r['prob'] >= 0.5 else an
                actual_winner = actual_results.get((hn, an))
                if not actual_winner:
                    continue
                
                hit = 1 if pick == actual_winner else 0
                
                conf_raw = r['conf_raw']
                conf_cal = r['conf_calibrated']
                cons = r['consensus']
                god = r['god_mode']
                
                # Tier logic (matches predict_live)
                if god and conf_cal >= 80:
                    tier = "GOD"
                elif conf_cal >= 78:
                    tier = "SNIPER"
                elif conf_cal >= 65:
                    tier = "VOLUMEN"
                else:
                    tier = "STANDARD"
                
                all_results.append({
                    'date': date_str, 'home': hn, 'away': an,
                    'pick': pick, 'prob': round(r['prob'], 4),
                    'conf': conf_cal, 'tier': tier,
                    'consensus': cons, 'actual_winner': actual_winner,
                    'hit': hit
                })
            
            # Write incremental CSV
            if all_results:
                pd.DataFrame(all_results).to_csv(csv_path, index=False)
            
            curr_test += timedelta(days=1)
        
        curr_train_start = test_end + timedelta(days=1)
    
    # ── RESULTS ──
    print(f"\n{'='*70}")
    print("  BACKTEST REAL — Walk-forward (sin override, cutoff matching)")
    print(f"{'='*70}")
    
    if not all_results:
        print("No results generated.")
        return
    
    df_res = pd.DataFrame(all_results)
    total = len(df_res)
    hits = df_res['hit'].sum()
    acc = hits / total if total > 0 else 0
    
    print(f"\n  Accuracy: {acc:.2%} ({hits}/{total})")
    
    for tier in ['GOD', 'SNIPER', 'VOLUMEN', 'STANDARD']:
        df_t = df_res[df_res['tier'] == tier]
        if len(df_t) > 0:
            acc_t = df_t['hit'].sum() / len(df_t)
            profit = (df_t['hit'].sum() * 1.91) - len(df_t)
            print(f"  {tier:10s}: {acc_t:.2%} ({df_t['hit'].sum():3d}/{len(df_t):3d}) | Profit: {profit:+.2f}u")
    
    profit_total = (hits * 1.91) - total
    print(f"\n  Total Profit: {profit_total:+.2f}u (at -110)")
    print(f"\nResultados guardados: backtest_real_results.csv")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2024-04-01')
    parser.add_argument('--end', default='2026-06-01')
    parser.add_argument('--weeks', type=int, default=2, help='Weeks per training session')
    args = parser.parse_args()
    run_backtest_real(args.start, args.end, weeks_per_train=args.weeks)
