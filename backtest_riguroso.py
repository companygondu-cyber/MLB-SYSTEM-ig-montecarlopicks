import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import sqlite3

# Import pipeline components
import pipeline
from models.game import train_multiwindow
from core_features import build_all_features

def get_actual_results(date_str):
    """Fetch actual game results for a specific date using statsapi."""
    try:
        import statsapi
        games = statsapi.schedule(date=date_str)
        results = {}
        for g in games:
            status = g.get('status', '')
            if 'Final' in status or 'Completed' in status:
                hn = g['home_name']
                an = g['away_name']
                hr = int(g.get('home_score', 0) or 0)
                ar = int(g.get('away_score', 0) or 0)
                winner = hn if hr > ar else an
                results[(hn, an)] = winner
        return results
    except Exception as e:
        print(f"Error fetching actual results for {date_str}: {e}")
        return {}

def run_backtest_for_date(target_date_str, skip_train=False):
    target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
    train_max_dt = target_dt - timedelta(days=1)
    train_max_str = train_max_dt.strftime('%Y-%m-%d')
    
    print(f"\n=======================================================")
    print(f"🔄 BACKTEST PARA EL DÍA: {target_date_str}")
    print(f"   [Corte de Entrenamiento: {train_max_str}]")
    print(f"=======================================================\n")
    
    if not skip_train:
        print(f"[1/3] Cargando DB hasta {train_max_str} para entrenamiento...")
        df_train, pp_train = pipeline.load_data(max_date=train_max_str)
        
        print(f"[2/3] Construyendo features (sin fuga de datos)...")
        # build_all_features naturally iterates through df_train which is already cut off
        df_train, feats, target = build_all_features(df_train, pp_train)
        
        print(f"[3/3] Entrenando Modelos y guardando pesos...")
        train_multiwindow(df_train, feats, target)
    else:
        print(f"Saltando entrenamiento (skip_train=True)...")

    print(f"\n[SCAN] Ejecutando predict_live para {target_date_str}...")
    # Ejecutamos predict_live (que a su vez usará DB hasta target_date_str - 1 internamente para las inferencias)
    predictions = pipeline.predict_live(date_str=target_date_str, skip_sync=True, skip_train=True)
    
    actual_results = get_actual_results(target_date_str)
    
    daily_stats = []
    
    if predictions:
        for p in predictions:
            home = p['home']
            away = p['away']
            pick = p['pick']
            tier = p['tier']
            conf = p['conf_cal']
            
            actual_winner = actual_results.get((home, away))
            if not actual_winner:
                # Sometimes names might mismatch slightly or game got postponed
                continue
                
            hit = 1 if pick == actual_winner else 0
            
            daily_stats.append({
                'date': target_date_str,
                'home': home,
                'away': away,
                'pick': pick,
                'tier': tier,
                'conf_cal': conf,
                'actual_winner': actual_winner,
                'hit': hit
            })
    return daily_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--skip_train', action='store_true', help='Skip training step (for debugging)')
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, '%Y-%m-%d')
    end_dt = datetime.strptime(args.end, '%Y-%m-%d')
    
    all_results = []
    
    curr_dt = start_dt
    
    # Initialize empty CSV
    pd.DataFrame(columns=['date', 'home', 'away', 'pick', 'tier', 'conf_cal', 'actual_winner', 'hit']).to_csv('backtest_results_rigorous.csv', index=False)
    
    last_trained_limit = None
    
    while curr_dt <= end_dt:
        date_str = curr_dt.strftime('%Y-%m-%d')
        target_dt = curr_dt
        train_max_dt = target_dt - timedelta(days=3)
        
        # Decide if we need to train
        if last_trained_limit is None or (train_max_dt - last_trained_limit).days >= 3:
            should_skip_train = args.skip_train
            # We will update last_trained_limit only if we actually train
            if not should_skip_train:
                last_trained_limit = train_max_dt
        else:
            should_skip_train = True
            
        print(f"\n[PROGRESS] Comenzando día {date_str} (skip_train={should_skip_train})...", flush=True)
        with open('backtest_progress.txt', 'w') as f:
            f.write(f"Procesando: {date_str}")
        
        daily_stats = run_backtest_for_date(date_str, skip_train=should_skip_train)
        
        if daily_stats:
            all_results.extend(daily_stats)
            # Append daily stats to CSV immediately
            pd.DataFrame(daily_stats).to_csv('backtest_results_rigorous.csv', mode='a', header=False, index=False)
            
        curr_dt += timedelta(days=1)
        
    print("\n\n=======================================================", flush=True)
    print(f"📊 RESULTADOS DEL BACKTEST RIGUROSO ({args.start} a {args.end})", flush=True)
    print("=======================================================\n", flush=True)
    
    if not all_results:
        print("No se generaron resultados (probablemente no hubo juegos o hubo un error).", flush=True)
        with open('backtest_progress.txt', 'w') as f:
            f.write(f"Done. No results.")
        return
        
    df_res = pd.DataFrame(all_results)
    
    total_games = len(df_res)
    total_hits = df_res['hit'].sum()
    acc = total_hits / total_games if total_games > 0 else 0
    
    print(f"Global: {acc:.2%} ({total_hits}/{total_games})", flush=True)
    
    # Por Tier
    for tier in df_res['tier'].unique():
        df_tier = df_res[df_res['tier'] == tier]
        hits = df_tier['hit'].sum()
        total = len(df_tier)
        acc_t = hits / total if total > 0 else 0
        profit = (hits * 1.91) - total
        print(f"Tier {tier:15s}: {acc_t:.2%} ({hits:2d}/{total:2d}) | Profit: {profit:+.2f}u", flush=True)
        
    with open('backtest_progress.txt', 'w') as f:
        f.write(f"Done. {total_games} games processed.")
    print("\nResultados detallados guardados en backtest_results_rigorous.csv", flush=True)

if __name__ == '__main__':
    main()
