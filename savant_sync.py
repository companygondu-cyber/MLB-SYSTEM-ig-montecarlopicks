import sys, os, sqlite3, warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# SSL Bypass for School DNS/Proxy
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
old_get = requests.get
requests.get = lambda *args, **kwargs: old_get(*args, **{**kwargs, 'verify': False})
old_post = requests.post
requests.post = lambda *args, **kwargs: old_post(*args, **{**kwargs, 'verify': False})

import pybaseball
from pybaseball import statcast

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'omega_2026_BETA.db')

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    conn.execute('PRAGMA journal_mode=WAL;')
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS savant_batter_daily (
        game_date TEXT,
        player_id INTEGER,
        player_name TEXT,
        p_throws TEXT,
        bbe INTEGER,
        barrels INTEGER,
        hard_hits INTEGER,
        avg_ev REAL,
        avg_xwoba REAL,
        avg_bat_speed REAL,
        avg_swing_length REAL,
        sweet_spot_count INTEGER,
        z_swings INTEGER,
        z_pitches INTEGER,
        o_swings INTEGER,
        o_pitches INTEGER,
        PRIMARY KEY (game_date, player_id, p_throws)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS savant_pitcher_daily (
        game_date TEXT,
        player_id INTEGER,
        player_name TEXT,
        stand TEXT,
        bbe INTEGER,
        barrels_allowed INTEGER,
        hard_hits_allowed INTEGER,
        avg_ev_allowed REAL,
        avg_xwoba_allowed REAL,
        avg_release_spin_rate REAL,
        avg_release_extension REAL,
        PRIMARY KEY (game_date, player_id, stand)
    )
    ''')
    conn.commit()
    conn.close()

def process_chunk(start_dt, end_dt):
    print(f"Fetching Statcast data from {start_dt} to {end_dt}...")
    try:
        df = statcast(start_dt=start_dt, end_dt=end_dt)
    except Exception as e:
        print(f"Error fetching data for range {start_dt} to {end_dt}: {e}")
        return
    
    if df is None or df.empty:
        print(f"No data found for period {start_dt} to {end_dt}.")
        return
 
    # Fix game_date format if needed
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')

    # Identify barrels and hard hits
    df['is_barrel'] = (df['launch_speed_angle'] == 6).fillna(False).astype(int)
    df['is_hard_hit'] = (df['launch_speed'] >= 95.0).fillna(False).astype(int)
    df['is_sweet_spot'] = ((df['launch_angle'] >= 8) & (df['launch_angle'] <= 32)).fillna(False).astype(int)
    
    # Calculate swing decisions
    swings = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 
              'in_play_hit_into_out', 'in_play_run_scored', 'in_play_no_out', 
              'foul_bunt', 'missed_bunt', 'swinging_strike_bunt']
    df['is_swing'] = df['description'].isin(swings).fillna(False).astype(int)
    df['in_zone'] = df['zone'].le(9).fillna(False).astype(int)
    
    df['is_o_swing'] = (((df['is_swing'] == 1) & (df['in_zone'] == 0))).astype(int)
    df['is_o_pitch'] = ((df['in_zone'] == 0) & df['zone'].notna()).astype(int)
    
    df['is_z_swing'] = (((df['is_swing'] == 1) & (df['in_zone'] == 1))).astype(int)
    df['is_z_pitch'] = ((df['in_zone'] == 1) & df['zone'].notna()).astype(int)
    
    # ─── AGGREGATE BATTERS ───
    b_agg = df.groupby(['game_date', 'batter', 'player_name', 'p_throws']).agg(
        bbe=('launch_speed', 'count'),
        barrels=('is_barrel', 'sum'),
        hard_hits=('is_hard_hit', 'sum'),
        avg_ev=('launch_speed', 'mean'),
        avg_xwoba=('estimated_woba_using_speedangle', 'mean'),
        bat_speed_sum=('bat_speed', 'sum'),
        bat_speed_count=('bat_speed', 'count'),
        swing_length_sum=('swing_length', 'sum'),
        swing_length_count=('swing_length', 'count'),
        sweet_spot_count=('is_sweet_spot', 'sum'),
        z_swings=('is_z_swing', 'sum'),
        z_pitches=('is_z_pitch', 'sum'),
        o_swings=('is_o_swing', 'sum'),
        o_pitches=('is_o_pitch', 'sum')
    ).reset_index()
    
    b_agg['avg_bat_speed'] = b_agg['bat_speed_sum'] / b_agg['bat_speed_count'].replace(0, np.nan)
    b_agg['avg_swing_length'] = b_agg['swing_length_sum'] / b_agg['swing_length_count'].replace(0, np.nan)
    b_agg.drop(columns=['bat_speed_sum', 'bat_speed_count', 'swing_length_sum', 'swing_length_count'], inplace=True)
    b_agg.rename(columns={'batter': 'player_id'}, inplace=True)
    
    # ─── AGGREGATE PITCHERS ───
    p_agg = df.groupby(['game_date', 'pitcher', 'player_name', 'stand']).agg(
        bbe=('launch_speed', 'count'),
        barrels_allowed=('is_barrel', 'sum'),
        hard_hits_allowed=('is_hard_hit', 'sum'),
        avg_ev_allowed=('launch_speed', 'mean'),
        avg_xwoba_allowed=('estimated_woba_using_speedangle', 'mean'),
        spin_sum=('release_spin_rate', 'sum'),
        spin_count=('release_spin_rate', 'count'),
        ext_sum=('release_extension', 'sum'),
        ext_count=('release_extension', 'count')
    ).reset_index()
    
    p_agg['avg_release_spin_rate'] = p_agg['spin_sum'] / p_agg['spin_count'].replace(0, np.nan)
    p_agg['avg_release_extension'] = p_agg['ext_sum'] / p_agg['ext_count'].replace(0, np.nan)
    p_agg.drop(columns=['spin_sum', 'spin_count', 'ext_sum', 'ext_count'], inplace=True)
    p_agg.rename(columns={'pitcher': 'player_id'}, inplace=True)
    
    # ─── SAVE TO DB ───
    conn = get_conn()
    try:
        b_records = b_agg.to_dict('records')
        p_records = p_agg.to_dict('records')
        
        cur = conn.cursor()
        for r in b_records:
            cur.execute("""
                INSERT OR REPLACE INTO savant_batter_daily 
                (game_date, player_id, player_name, p_throws, bbe, barrels, hard_hits, avg_ev, avg_xwoba,
                 avg_bat_speed, avg_swing_length, sweet_spot_count, z_swings, z_pitches, o_swings, o_pitches)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (r['game_date'], r['player_id'], r['player_name'], r['p_throws'], r['bbe'], r['barrels'], r['hard_hits'], r['avg_ev'], r['avg_xwoba'],
                  r['avg_bat_speed'], r['avg_swing_length'], r['sweet_spot_count'], r['z_swings'], r['z_pitches'], r['o_swings'], r['o_pitches']))
            
        for r in p_records:
            cur.execute("""
                INSERT OR REPLACE INTO savant_pitcher_daily 
                (game_date, player_id, player_name, stand, bbe, barrels_allowed, hard_hits_allowed, avg_ev_allowed, avg_xwoba_allowed,
                 avg_release_spin_rate, avg_release_extension)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (r['game_date'], r['player_id'], r['player_name'], r['stand'], r['bbe'], r['barrels_allowed'], r['hard_hits_allowed'], r['avg_ev_allowed'], r['avg_xwoba_allowed'],
                  r['avg_release_spin_rate'], r['avg_release_extension']))
        conn.commit()
    except Exception as e:
        print(f"Error saving to DB for range {start_dt} to {end_dt}: {e}")
    finally:
        conn.close()
    
    print(f"Processed and saved {len(b_agg)} batter-games and {len(p_agg)} pitcher-games ({start_dt} to {end_dt}).")

def run_bootstrap():
    init_db()
    seasons = [
        ('2023-03-30', '2023-10-31'),
        ('2024-03-20', '2024-10-31'),
        ('2025-03-20', '2025-10-31'),
        ('2026-03-20', '2026-06-01')
    ]
    chunks = []
    for start_s, end_s in seasons:
        dt_start = datetime.strptime(start_s, '%Y-%m-%d')
        dt_end = datetime.strptime(end_s, '%Y-%m-%d')
        
        while dt_start <= dt_end:
            chunk_end = dt_start + timedelta(days=14)
            if chunk_end > dt_end:
                chunk_end = dt_end
            
            chunks.append((dt_start.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
            dt_start = chunk_end + timedelta(days=1)
            
    print(f"Starting concurrent bootstrap with {len(chunks)} chunks...")
    with ThreadPoolExecutor(max_workers=6) as executor: # Use 6 workers to avoid overloading or rate limiting
        futures = {executor.submit(process_chunk, c[0], c[1]): c for c in chunks}
        for future in as_completed(futures):
            c = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error in future for chunk {c[0]} to {c[1]}: {e}")

def sync_daily():
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT MAX(game_date) FROM savant_pitcher_daily")
    row = cur.fetchone()
    last_date = row[0] if row and row[0] else '2026-03-20'
    conn.close()
    
    dt_last = datetime.strptime(last_date, '%Y-%m-%d')
    dt_today = datetime.now()
    
    if dt_last.date() >= dt_today.date():
        print("Savant data is already up to date.")
        return
        
    start_str = (dt_last + timedelta(days=1)).strftime('%Y-%m-%d')
    end_str = dt_today.strftime('%Y-%m-%d')
    print(f"Syncing daily Savant from {start_str} to {end_str}")
    process_chunk(start_str, end_str)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--bootstrap':
        print("Starting Statcast Bootstrap (Concurrent Mode)...")
        run_bootstrap()
        print("Bootstrap complete!")
    else:
        sync_daily()
