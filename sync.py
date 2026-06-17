"""OmegaFinal-MLB data sync. Single module for: MLB schedule + boxscores + Statcast + lineups + schema."""

import os
import sys
import sqlite3
import warnings
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import urllib3

# ── SSL bypass for school DNS/proxy (preserved from original) ──
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_old_get = requests.get
requests.get = lambda *args, **kwargs: _old_get(*args, **{**kwargs, 'verify': False})
_old_post = requests.post
requests.post = lambda *args, **kwargs: _old_post(*args, **{**kwargs, 'verify': False})

warnings.filterwarnings('ignore')

import pybaseball
from pybaseball import statcast

import statsapi

from config import CONFIG
from db import DBConnection

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Schema initialization
# ════════════════════════════════════════════════════════════════════════════

def init_schema() -> None:
    """Create the Savant daily tables and ensure migratable columns exist."""
    db = DBConnection()
    conn = sqlite3.connect(db.path, timeout=60.0)
    conn.execute('PRAGMA journal_mode=WAL;')
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

    _migrate_savant_columns()


def _migrate_savant_columns() -> None:
    """Add columns that may be missing on older DBs (idempotent)."""
    db = DBConnection()
    conn = sqlite3.connect(db.path)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(savant_batter_daily)")
    batter_cols = [c[1] for c in cur.fetchall()]

    cur.execute("PRAGMA table_info(savant_pitcher_daily)")
    pitcher_cols = [c[1] for c in cur.fetchall()]

    new_batter_cols = {
        'avg_bat_speed': 'REAL', 'avg_swing_length': 'REAL', 'sweet_spot_count': 'INTEGER',
        'z_swings': 'INTEGER', 'z_pitches': 'INTEGER', 'o_swings': 'INTEGER', 'o_pitches': 'INTEGER',
    }
    new_pitcher_cols = {
        'avg_release_spin_rate': 'REAL', 'avg_release_extension': 'REAL',
    }

    for col, ctype in new_batter_cols.items():
        if col not in batter_cols:
            cur.execute(f"ALTER TABLE savant_batter_daily ADD COLUMN {col} {ctype}")
    for col, ctype in new_pitcher_cols.items():
        if col not in pitcher_cols:
            cur.execute(f"ALTER TABLE savant_pitcher_daily ADD COLUMN {col} {ctype}")

    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# Statcast (Savant) sync
# ════════════════════════════════════════════════════════════════════════════

def _process_savant_chunk(start_dt: str, end_dt: str) -> None:
    """Fetch + aggregate Statcast data for a date range and write to DB."""
    print(f"Fetching Statcast data from {start_dt} to {end_dt}...")
    try:
        df = statcast(start_dt=start_dt, end_dt=end_dt)
    except Exception as e:
        print(f"Error fetching data for range {start_dt} to {end_dt}: {e}")
        return

    if df is None or df.empty:
        print(f"No data found for period {start_dt} to {end_dt}.")
        return

    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    df['is_barrel'] = (df['launch_speed_angle'] == 6).fillna(False).astype(int)
    df['is_hard_hit'] = (df['launch_speed'] >= 95.0).fillna(False).astype(int)
    df['is_sweet_spot'] = ((df['launch_angle'] >= 8) & (df['launch_angle'] <= 32)).fillna(False).astype(int)

    swings = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip',
              'in_play_hit_into_out', 'in_play_run_scored', 'in_play_no_out',
              'foul_bunt', 'missed_bunt', 'swinging_strike_bunt']
    df['is_swing'] = df['description'].isin(swings).fillna(False).astype(int)
    df['in_zone'] = df['zone'].le(9).fillna(False).astype(int)
    df['is_o_swing'] = ((df['is_swing'] == 1) & (df['in_zone'] == 0)).astype(int)
    df['is_o_pitch'] = ((df['in_zone'] == 0) & df['zone'].notna()).astype(int)
    df['is_z_swing'] = ((df['is_swing'] == 1) & (df['in_zone'] == 1)).astype(int)
    df['is_z_pitch'] = ((df['in_zone'] == 1) & df['zone'].notna()).astype(int)

    b_agg = df.groupby(['game_date', 'batter', 'player_name', 'p_throws']).agg(
        bbe=('launch_speed', 'count'), barrels=('is_barrel', 'sum'),
        hard_hits=('is_hard_hit', 'sum'), avg_ev=('launch_speed', 'mean'),
        avg_xwoba=('estimated_woba_using_speedangle', 'mean'),
        bat_speed_sum=('bat_speed', 'sum'), bat_speed_count=('bat_speed', 'count'),
        swing_length_sum=('swing_length', 'sum'), swing_length_count=('swing_length', 'count'),
        sweet_spot_count=('is_sweet_spot', 'sum'),
        z_swings=('is_z_swing', 'sum'), z_pitches=('is_z_pitch', 'sum'),
        o_swings=('is_o_swing', 'sum'), o_pitches=('is_o_pitch', 'sum'),
    ).reset_index()

    b_agg['avg_bat_speed'] = b_agg['bat_speed_sum'] / b_agg['bat_speed_count'].replace(0, np.nan)
    b_agg['avg_swing_length'] = b_agg['swing_length_sum'] / b_agg['swing_length_count'].replace(0, np.nan)
    b_agg.drop(columns=['bat_speed_sum', 'bat_speed_count', 'swing_length_sum', 'swing_length_count'], inplace=True)
    b_agg.rename(columns={'batter': 'player_id'}, inplace=True)

    p_agg = df.groupby(['game_date', 'pitcher', 'player_name', 'stand']).agg(
        bbe=('launch_speed', 'count'), barrels_allowed=('is_barrel', 'sum'),
        hard_hits_allowed=('is_hard_hit', 'sum'),
        avg_ev_allowed=('launch_speed', 'mean'),
        avg_xwoba_allowed=('estimated_woba_using_speedangle', 'mean'),
        spin_sum=('release_spin_rate', 'sum'), spin_count=('release_spin_rate', 'count'),
        ext_sum=('release_extension', 'sum'), ext_count=('release_extension', 'count'),
    ).reset_index()

    p_agg['avg_release_spin_rate'] = p_agg['spin_sum'] / p_agg['spin_count'].replace(0, np.nan)
    p_agg['avg_release_extension'] = p_agg['ext_sum'] / p_agg['ext_count'].replace(0, np.nan)
    p_agg.drop(columns=['spin_sum', 'spin_count', 'ext_sum', 'ext_count'], inplace=True)
    p_agg.rename(columns={'pitcher': 'player_id'}, inplace=True)

    db = DBConnection()
    conn = sqlite3.connect(db.path, timeout=60.0)
    try:
        cur = conn.cursor()
        for r in b_agg.to_dict('records'):
            cur.execute("""
                INSERT OR REPLACE INTO savant_batter_daily
                (game_date, player_id, player_name, p_throws, bbe, barrels, hard_hits, avg_ev, avg_xwoba,
                 avg_bat_speed, avg_swing_length, sweet_spot_count, z_swings, z_pitches, o_swings, o_pitches)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (r['game_date'], r['player_id'], r['player_name'], r['p_throws'],
                  r['bbe'], r['barrels'], r['hard_hits'], r['avg_ev'], r['avg_xwoba'],
                  r['avg_bat_speed'], r['avg_swing_length'], r['sweet_spot_count'],
                  r['z_swings'], r['z_pitches'], r['o_swings'], r['o_pitches']))
        for r in p_agg.to_dict('records'):
            cur.execute("""
                INSERT OR REPLACE INTO savant_pitcher_daily
                (game_date, player_id, player_name, stand, bbe, barrels_allowed, hard_hits_allowed,
                 avg_ev_allowed, avg_xwoba_allowed, avg_release_spin_rate, avg_release_extension)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (r['game_date'], r['player_id'], r['player_name'], r['stand'],
                  r['bbe'], r['barrels_allowed'], r['hard_hits_allowed'],
                  r['avg_ev_allowed'], r['avg_xwoba_allowed'],
                  r['avg_release_spin_rate'], r['avg_release_extension']))
        conn.commit()
    finally:
        conn.close()

    print(f"Processed and saved {len(b_agg)} batter-games and {len(p_agg)} pitcher-games ({start_dt} to {end_dt}).")


def sync_savant_bootstrap() -> None:
    """One-time full historical Statcast sync (2023-2026)."""
    init_schema()
    seasons = [
        ('2023-03-30', '2023-10-31'),
        ('2024-03-20', '2024-10-31'),
        ('2025-03-20', '2025-10-31'),
        ('2026-03-20', '2026-06-01'),
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
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_process_savant_chunk, c[0], c[1]): c for c in chunks}
        for future in as_completed(futures):
            c = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error in future for chunk {c[0]} to {c[1]}: {e}")


def sync_savant_daily() -> None:
    """Daily incremental Statcast sync: from last recorded date to today."""
    init_schema()
    db = DBConnection()
    conn = sqlite3.connect(db.path)
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
    _process_savant_chunk(start_str, end_str)


# ════════════════════════════════════════════════════════════════════════════
# MLB schedule / boxscore sync (from omega_v3:sync_data)
# ════════════════════════════════════════════════════════════════════════════

def sync_mlb_schedule(days_back: int = 2) -> int:
    """Sync MLB schedule for the last N days. Returns number of new games."""
    target_dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                    for i in range(days_back + 1)]
    new_count = 0
    db = DBConnection()
    for date_str in target_dates:
        try:
            games = statsapi.schedule(date=date_str)
        except Exception as e:
            logger.warning(f"Could not fetch schedule for {date_str}: {e}")
            continue
        for g in games:
            game_id = g.get('game_id')
            if not game_id:
                continue
            try:
                db.execute("""
                    INSERT OR IGNORE INTO historico_partidos
                    (game_pk, date, home_team, away_team, venue, series_game_number,
                     pp_h_starter_name, pp_h_starter_id, pp_a_starter_name, pp_a_starter_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_id, date_str, g.get('home_name', ''), g.get('away_name', ''),
                      g.get('venue_name', ''), g.get('game_num', 1),
                      g.get('home_probable_pitcher', ''), 0,
                      g.get('away_probable_pitcher', ''), 0))
                new_count += 1
            except Exception as e:
                logger.warning(f"Failed to insert game {game_id}: {e}")
    return new_count


# ════════════════════════════════════════════════════════════════════════════
# Lineup prediction (from lineup_predictor.LineupPredictor)
# ════════════════════════════════════════════════════════════════════════════

class LineupPredictor:
    """Markov-based lineup predictor using recent games as priors.

    Preserved from lineup_predictor.py verbatim — only the import paths changed.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or str(CONFIG.paths.db)
        self._stats_cache = {}
        self._team_history = {}

    def _load_team(self, team_id):
        if team_id in self._team_history:
            return
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT date, player_id, player_name, batting_order
            FROM batter_performances
            WHERE team_id = ? AND date >= '2025-01-01'
            ORDER BY date, batting_order
        """, (team_id,)).fetchall()
        conn.close()
        games = defaultdict(list)
        for rdate, pid, pname, b_order in rows:
            slot = int(b_order) // 100
            if 1 <= slot <= 9:
                games[rdate].append((slot, pid, pname))
        for d in games:
            games[d].sort(key=lambda x: x[0])
        self._team_history[team_id] = sorted(games.items(), key=lambda x: x[0])

    def predict_lineup(self, team_id, date_str):
        self._load_team(team_id)
        games = self._team_history.get(team_id, [])
        if not games:
            return None
        past = [(d, l) for d, l in games if d[:10] < date_str]
        if len(past) < 2:
            return None

        # Fetch live injury list (only when predicting for today)
        injured_ids = set()
        is_today = (date_str == datetime.now().strftime('%Y-%m-%d'))
        if is_today:
            try:
                import requests as _req
                url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=injury"
                resp = _req.get(url, timeout=2.0, verify=False)
                if resp.status_code == 200:
                    roster_data = resp.json().get('roster', [])
                    for player in roster_data:
                        p_id = player.get('person', {}).get('id')
                        if p_id:
                            injured_ids.add(int(p_id))
            except Exception as e:
                logger.debug(f"Could not fetch injury list for team {team_id}: {e}")

        last_date, last_lineup = past[-1]
        last_map = {slot: pid for slot, pid, _ in last_lineup}
        last_pids = set(last_map.values())

        # Build per-player profiles from games in last 14 days
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        player_apps = defaultdict(int)
        player_slots = defaultdict(lambda: Counter())
        recent_games = [g for g in past if datetime.strptime(g[0][:10], '%Y-%m-%d') >= date_obj - timedelta(days=14)]
        if not recent_games:
            recent_games = past[-9:]

        for gdate, lineup in recent_games:
            g_obj = datetime.strptime(gdate[:10], '%Y-%m-%d')
            diff_days = (date_obj - g_obj).days
            if diff_days <= 3: weight = 3
            elif diff_days <= 7: weight = 2
            else: weight = 1

            seen = set()
            for slot, pid, _ in lineup:
                if pid not in seen:
                    if pid not in injured_ids:
                        player_apps[pid] += weight
                        seen.add(pid)
                player_slots[pid][slot] += weight

        player_rank = sorted(player_apps.items(), key=lambda x: (-x[1], x[0]))
        top_9_pids = [pid for pid, _ in player_rank[:9]]
        top_9_set = set(top_9_pids)

        assigned = {}
        used_pids = set()

        # Phase 1: Keep last game slots for returning starters
        for slot in range(1, 10):
            pid = last_map.get(slot)
            if pid and pid in top_9_set:
                assigned[slot] = pid
                used_pids.add(pid)

        # Phase 2: For remaining players, match to modal slot
        remaining_pids = [pid for pid in top_9_pids if pid not in used_pids]
        empty_slots = [s for s in range(1, 10) if s not in assigned]

        for pid in list(remaining_pids):
            mode_slot = player_slots[pid].most_common(1)[0][0] if player_slots[pid] and sum(player_slots[pid].values()) > 0 else None
            if mode_slot and mode_slot in empty_slots:
                assigned[mode_slot] = pid
                used_pids.add(pid)
                remaining_pids.remove(pid)
                empty_slots.remove(mode_slot)

        # Phase 3: Remaining unassigned to remaining empty slots
        for pid in remaining_pids:
            if empty_slots:
                best_slot = None
                best_aff = -1
                for s in empty_slots:
                    aff = player_slots[pid].get(s, 0) / max(player_apps[pid], 1)
                    if aff > best_aff:
                        best_aff = aff
                        best_slot = s
                if best_slot:
                    assigned[best_slot] = pid
                    used_pids.add(pid)
                    empty_slots.remove(best_slot)

        # Phase 4: Fill any truly empty slots
        for slot in range(1, 10):
            if slot not in assigned:
                for pid in top_9_pids:
                    if pid not in used_pids:
                        assigned[slot] = pid
                        used_pids.add(pid)
                        break
        for slot in range(1, 10):
            if slot not in assigned:
                assigned[slot] = last_map.get(slot, top_9_pids[0] if top_9_pids else 0)

        name_lookup = {}
        for gdate, lineup in past:
            for s, p, n in lineup:
                if p not in name_lookup:
                    name_lookup[p] = n

        result = []
        for slot in range(1, 10):
            pid = assigned.get(slot, 0)
            name = name_lookup.get(pid, 'Unknown')
            result.append({'personId': pid, 'name': name})
        return result

    def backtest(self, start_date='2026-05-01', end_date='2026-05-24'):
        conn = sqlite3.connect(self.db_path)
        games = conn.execute("""
            SELECT DISTINCT bp.game_pk, bp.date, bp.team_id
            FROM batter_performances bp
            WHERE bp.date >= ? AND bp.date < ?
            ORDER BY bp.date
        """, (start_date, end_date)).fetchall()
        conn.close()
        correct_slots = 0; total_slots = 0
        correct_who = 0; total_who = 0
        games_tested = 0; games_with_pred = 0; exact_9 = 0
        seen = set()
        for gpk, gdate, tid in games:
            key = (gpk, tid)
            if key in seen:
                continue
            seen.add(key)
            conn = sqlite3.connect(self.db_path)
            actual = conn.execute("""
                SELECT player_id, player_name, batting_order
                FROM batter_performances
                WHERE game_pk = ? AND team_id = ? AND (batting_order % 100) = 0
                ORDER BY batting_order
            """, (gpk, tid)).fetchall()
            conn.close()
            if len(actual) < 9:
                continue
            games_tested += 1
            predicted = self.predict_lineup(tid, gdate[:10])
            if predicted is None or len(predicted) < 9:
                continue
            games_with_pred += 1
            actual_order = {int(r[2])//100: r[0] for r in actual}
            actual_set = set(r[0] for r in actual)
            pred_set = set(b['personId'] for b in predicted)
            if pred_set == actual_set:
                exact_9 += 1
            for slot, apid in actual_order.items():
                total_slots += 1
                if predicted[slot-1]['personId'] == apid:
                    correct_slots += 1
            for b in predicted:
                total_who += 1
                if b['personId'] in actual_set:
                    correct_who += 1
        return {
            'games_tested': games_tested, 'games_with_prediction': games_with_pred,
            'who_accuracy': correct_who / max(total_who, 1),
            'slot_accuracy': correct_slots / max(total_slots, 1),
            'exact_9': exact_9,
            'correct_slots': correct_slots, 'total_slots': total_slots,
            'correct_who': correct_who, 'total_who': total_who,
        }
