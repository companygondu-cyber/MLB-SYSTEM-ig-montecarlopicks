#!/usr/bin/env python3
"""OMEGA v3 — UN solo sistema. 85 features + 4 ventanas × 5 modelos + calibración real."""
import os,sys,sqlite3,warnings,argparse,json,shutil,logging,ssl,requests,csv
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
_orig_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    if 'headers' not in kwargs or kwargs['headers'] is None:
        kwargs['headers'] = {}
    kwargs['headers']['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    return _orig_request(self, method, url, **kwargs)
requests.Session.request = patched_request
import numpy as np, pandas as pd, joblib
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
from datetime import datetime,date,timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR,'data','omega_2026_BETA.db')
DB_BACKUP = DB_PATH+'.backup'
MOMIO_CACHE = os.path.join(BASE_DIR,'data','prescan_cache.json')
sys.path.insert(0, BASE_DIR)
from core_features import build_all_features, TEAM_IDS, TEAM_DIV, PARK_FACTORS, DB_PATH as FEATURES_DB_PATH
from config import USE_V3_ENSEMBLE
from core_training import (train_multiwindow, load_all_models, ensemble_predict,
                             models_exist as _legacy_models_exist, needs_retrain,
                             get_models_dir as MODELS_DIR)

from lineup_predictor import LineupPredictor
# Legacy sync imports removed
from savant_sync import sync_daily as sync_savant_daily


_lineup_predictor = None
def get_lineup_predictor():
    global _lineup_predictor
    if _lineup_predictor is None:
        _lineup_predictor = LineupPredictor()
    return _lineup_predictor

C={"B":"\033[1m","C":"\033[96m","G":"\033[92m","Y":"\033[93m",
   "R":"\033[91m","D":"\033[2m","X":"\033[0m","P":"\033[95m"}

def parse_baseball_ip(ip_val):
    if ip_val is None or pd.isna(ip_val): return 0.0
    try:
        ip_str = str(ip_val).strip()
        if '.' in ip_str:
            parts = ip_str.split('.')
            innings = int(parts[0])
            outs = int(parts[1])
            return innings + (outs / 3.0)
        else:
            return float(ip_str)
    except:
        return float(ip_val)

# ── MOMIO CHANGE: track pre-scan vs guess-scan vs confirmed ──
def load_prescan_cache():
    if os.path.exists(MOMIO_CACHE):
        try:
            with open(MOMIO_CACHE) as f: return json.load(f)
        except: pass
    return {}

def save_prescan_cache(data):
    os.makedirs(os.path.dirname(MOMIO_CACHE), exist_ok=True)
    with open(MOMIO_CACHE, 'w') as f: json.dump(data, f, indent=2)

def get_team_lineup_quality(conn, team_id, date_str, player_ids=None):
    """Calculate lineup quality score (avg TB/game) for a set of players.
    If player_ids is None, uses team's season average (baseline)."""
    c = conn.cursor()
    # Get team baseline first (used as fallback)
    team_row = c.execute("""
        SELECT AVG(pgs.tb), AVG(pgs.hits), AVG(pgs.hr)
        FROM player_game_stats pgs
        JOIN batter_performances bp ON pgs.game_pk = bp.game_pk AND pgs.player_id = bp.player_id
        WHERE bp.team_id = ? AND pgs.date < ?
    """, (team_id, date_str)).fetchone()
    team_baseline = {'tb': team_row[0] or 1.35, 'hits': team_row[1] or 0.75, 'hr': team_row[2] or 0.12}

    if player_ids:
        placeholders = ','.join(['?'] * len(player_ids))
        rows = c.execute(f"""
            SELECT AVG(pgs.tb), AVG(pgs.hits), AVG(pgs.hr), COUNT(*)
            FROM player_game_stats pgs
            WHERE pgs.player_id IN ({placeholders})
            AND pgs.date < ?
        """, (*player_ids, date_str)).fetchone()
        # If fewer than 5 players have data, fall back to team baseline
        if rows and rows[3] is not None and rows[3] >= 5 and rows[0] is not None:
            return {'tb': rows[0], 'hits': rows[1], 'hr': rows[2]}
        return team_baseline
    return team_baseline

def estimate_lineup_quality_delta(conn, team_name, team_id, date_str, lineup_batters):
    """Compare predicted lineup quality vs team average.
    Returns quality_diff percentage (positive = stronger lineup)."""
    if not lineup_batters or len(lineup_batters) < 5:
        return 0

    # Get player IDs from lineup
    player_ids = []
    for b in lineup_batters:
        pid = b.get('personId', 0) if isinstance(b, dict) else (b.get('player_id', 0) if isinstance(b, dict) else 0)
        if pid > 0:
            player_ids.append(pid)

    if len(player_ids) < 5:
        return 0

    # Predicted lineup quality
    pred_q = get_team_lineup_quality(conn, team_id, date_str, player_ids)
    # Team baseline (all batters)
    team_q = get_team_lineup_quality(conn, team_id, date_str, None)

    if team_q['tb'] <= 0:
        return 0

    # Quality diff: positive = lineup is BETTER than average
    tb_diff = (pred_q['tb'] - team_q['tb']) / team_q['tb'] * 100
    hr_diff = (pred_q['hr'] - team_q['hr']) / max(team_q['hr'], 0.01) * 100

    # Weighted: TB matters more, HR is bonus power signal
    return tb_diff * 0.7 + hr_diff * 0.3

def estimate_momio_change(game_key, conf_cal, scan, lineup_confirmed, hp, ap, hs, ast,
                          team_name=None, team_id=None, date_str=None,
                          lineup_batters=None):
    """Track momio shift across PRE-SCAN → GUESS-SCAN → CONFIRMED.
    Returns (pct_change, display_string)."""
    cache = load_prescan_cache()
    today = datetime.now().strftime('%Y-%m-%d')

    # ── CONFIRMED: show checkmark, compare actual vs predicted ──
    if lineup_confirmed and 'CONFIRMED' in scan:
        if game_key in cache:
            old = cache[game_key]
            prev_scan = old.get('scan', '')
            if 'GUESS' in prev_scan:
                delta = conf_cal - old['conf_cal']
                emoji = "📈" if delta > 0.5 else ("📉" if delta < -0.5 else "✅")
                del cache[game_key]
                save_prescan_cache(cache)
                return delta, f"✅ {emoji}"
            del cache[game_key]
            save_prescan_cache(cache)
        return 0, "✅"

    # ── GUESS-SCAN: evaluate lineup quality vs team average ──
    if 'GUESS' in scan:
        expected_delta = 0
        if team_name and team_id and date_str and lineup_batters:
            try:
                _conn = sqlite3.connect(DB_PATH)
                quality_pct = estimate_lineup_quality_delta(
                    _conn, team_name, team_id, date_str, lineup_batters)
                _conn.close()
                expected_delta = quality_pct * 0.5
                h_era = hs.get('era', 4.20) if hs else 4.20
                a_era = ast.get('era', 4.20) if ast else 4.20
                vol = 0
                if h_era > 5.0: vol += 0.8
                elif h_era > 3.5: vol += 0.4
                if a_era > 5.0: vol += 0.8
                elif a_era > 3.5: vol += 0.4
                expected_delta += vol if expected_delta > 0 else -vol
            except Exception as e:
                logger.debug(f"Momio quality calc failed: {e}")

        # Store for later CONFIRMED comparison
        cache[game_key] = {'conf_cal': conf_cal, 'date': today, 'scan': scan}
        save_prescan_cache(cache)

        if expected_delta > 0.5:
            return expected_delta, f"📈 +{expected_delta:.1f}%"
        elif expected_delta < -0.5:
            return expected_delta, f"📉 {expected_delta:.1f}%"
        else:
            return expected_delta, f"➡️ ~0%"

    # ── PRE-SCAN: estimate based on pitcher volatility only ──
    cache[game_key] = {'conf_cal': conf_cal, 'date': today, 'scan': scan}
    save_prescan_cache(cache)

    h_era = hs.get('era', 4.20) if hs else 4.20
    a_era = ast.get('era', 4.20) if ast else 4.20
    vol = 0
    if h_era > 5.0: vol += 1.5
    elif h_era > 3.5: vol += 0.8
    if a_era > 5.0: vol += 1.5
    elif a_era > 3.5: vol += 0.8
    est = 2.5 + vol
    if hp == 'TBD' or ap == 'TBD': est += 3.0
    if conf_cal >= 80: est *= 0.7
    elif conf_cal < 60: est *= 1.3
    return round(est, 1), f"⏳ ~{est:+.1f}%"

def backup_db():
    """Always create fresh backup before any write operation."""
    try:
        if os.path.exists(DB_PATH):
            shutil.copy2(DB_PATH, DB_BACKUP)
    except Exception as e:
        print(f"⚠️ Backup failed: {e}")

def restore_db():
    try:
        if os.path.exists(DB_BACKUP): shutil.copy2(DB_BACKUP, DB_PATH)
    except Exception as e: logging.warning(e)

def validate_db_schema():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_role ON pitcher_performances(role)")
        conn.commit()
        logger.info("Verificación de esquema completada: Índices confirmados.")
    except Exception as e: logger.error(f"Error al verificar el esquema de la DB: {e}")
    conn.close()

def sync_data(days_back=30):
    validate_db_schema()
    try: import statsapi
    except: return False
    print(f"{C['C']}Sync {days_back} días...{C['X']}")
    conn=sqlite3.connect(DB_PATH)
    existing=set(pd.read_sql("SELECT game_pk FROM historico_partidos",conn)['game_pk'].tolist())
    nuevo=0
    for i in range(days_back,-1,-1):
        d=(datetime.now()-timedelta(days=i)).strftime('%Y-%m-%d')
        try: games=statsapi.schedule(date=d)
        except Exception as e:
            logging.warning(e)
            continue
        for g in games:
            pk=g['game_id']; status=g.get('status','')
            # Fix #45: broader status matching
            if not any(status.startswith(s) for s in ('Final','Completed')) or pk in existing: continue
            try: box=statsapi.boxscore_data(pk)
            except Exception as e:
                logger.warning(f"Failed to fetch boxscore for game {pk}: {e}")
                continue
            hn,an=g['home_name'],g['away_name']
            hr=int(g.get('home_score',0)or 0); ar=int(g.get('away_score',0)or 0)
            hs,asc=box.get('home'),box.get('away')
            if not hs or not asc: continue
            # Extract team batting stats from boxscore
            h_bat=hs.get('teamStats',{}).get('batting',{})
            a_bat=asc.get('teamStats',{}).get('batting',{})
            h_ops_v=_safe_float(h_bat.get('ops'), 0.700)
            a_ops_v=_safe_float(a_bat.get('ops'), 0.700)
            h_avg_v=_safe_float(h_bat.get('avg'), 0.250)
            a_avg_v=_safe_float(a_bat.get('avg'), 0.250)
            try:
                conn.execute("""INSERT OR IGNORE INTO historico_partidos
                    (game_pk,date,home_team,away_team,venue,series_game_number,
                     h_runs_total,a_runs_total,h_hits_total,a_hits_total,
                     h_errors_total,a_errors_total,winner,total_innings,total_pitches,
                     h_ops,a_ops,h_avg,a_avg) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (pk,d,hn,an,g.get('venue',''),int(g.get('game_num',1)),
                     hr,ar,int(hs.get('hits',0)),int(asc.get('hits',0)),
                     int(hs.get('errors',0)),int(asc.get('errors',0)),
                     1 if hr>ar else 0,int(hs.get('inning',9)),
                     int(hs.get('pitches',0))+int(asc.get('pitches',0)),
                     h_ops_v,a_ops_v,h_avg_v,a_avg_v))
            except Exception as e: logging.warning(e)
            # Extract individual batter stats from boxscore
            _extract_batters(conn, box, pk, d)
            # Extract pitcher performances
            for side,tn in[('away',an),('home',hn)]:
                tid=int(g.get(f'{side}_id',0)); first=True
                for p in box.get(f'{side}Pitchers',[]):
                    pid=int(p.get('personId',0))
                    if pid<=0: continue
                    # Fix #36: safe ip parsing — 0.0 float won't be replaced by '0'
                    _ip_raw = p.get('ip', 0)
                    ip = max(float(_ip_raw) if _ip_raw is not None and _ip_raw != '' else 0.0, 0.1)
                    er=int(p.get('er',0) or 0)
                    h_val=int(p.get('h',0) or 0); bb_val=int(p.get('bb',0) or 0)
                    bf=int(p.get('battersFaced',0) or 0)  # Fix #12
                    try:
                        conn.execute("""INSERT OR IGNORE INTO pitcher_performances
                            (game_pk,date,team_id,player_id,player_name,role,
                             ip,hits,runs,er,k,bb,hr,whip,era_game,batters_faced)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (pk,d,tid,pid,p.get('name',''),
                             'starter' if first else 'reliever',
                             ip,h_val,int(p.get('r',0) or 0),er,
                             int(p.get('k',0) or 0),bb_val,
                             int(p.get('hr',0) or 0),
                             round((h_val+bb_val)/ip,2),
                             round(er*9/ip,2),bf))
                        first=False
                    except Exception as e:
                        logger.warning(f"Failed to insert pitcher {pid} for game {pk}: {e}")
            conn.commit(); nuevo+=1
    # Enrich starter stats — Fix #35: 0.0 ERA is valid for shutouts. Now also stores strikeout_rate/walk_rate.
    gids=conn.execute("SELECT hp.game_pk,hp.home_team,hp.away_team FROM historico_partidos hp WHERE (hp.h_starter_era IS NULL OR hp.h_starter_strikeout_rate IS NULL) AND hp.game_pk IN (SELECT DISTINCT game_pk FROM pitcher_performances WHERE role='starter')").fetchall()
    for pk,home,away in gids:
        try:
            for row in conn.execute("SELECT team_id,era_game,whip,ip,k,hits,bb FROM pitcher_performances WHERE game_pk=? AND role='starter'",(pk,)).fetchall():
                tid,era,whip,ip,k,ha,bb = row
                era_v = float(era) if era is not None else 4.0
                whip_v = float(whip) if whip is not None else 1.3
                bf = max(3*float(ip)+float(ha)+float(bb), 1)
                sr = round(float(k)/bf, 6) if k else 0.0
                wr = round(float(bb)/bf, 6) if bb else 0.0
                if TEAM_IDS.get(home)==tid:
                    conn.execute("UPDATE historico_partidos SET h_starter_era=?,h_starter_whip=?,h_starter_strikeout_rate=?,h_starter_walk_rate=? WHERE game_pk=?",(era_v,whip_v,sr,wr,pk))
                elif TEAM_IDS.get(away)==tid:
                    conn.execute("UPDATE historico_partidos SET a_starter_era=?,a_starter_whip=?,a_starter_strikeout_rate=?,a_starter_walk_rate=? WHERE game_pk=?",(era_v,whip_v,sr,wr,pk))
            conn.commit()
        except Exception as e:
            print(f"⚠️ Enrich starter {pk}: {e}")
    # Enrich games missing batter data retroactively
    _enrich_missing_batters(conn)
    conn.close()
    if nuevo: print(f"{C['G']}{nuevo} juegos nuevos{C['X']}")
    return nuevo>0

def _safe_float(val, default):
    """Safely parse float from API, handling empty strings and None. Fix ⚪ OPS=0 bug."""
    if val is None or pd.isna(val): return default
    try:
        f = float(val)
        return f  # Keep 0.0 as 0.0 — a real .000 stat is meaningful
    except (ValueError, TypeError):
        return default

def _extract_batters(conn, box, pk, d):
    """Extract individual batter stats from boxscore. Uses parameterized SQL (fix 🔴4)."""
    updates = {}
    try:
        conn.execute("DELETE FROM player_game_stats WHERE game_pk=?", (pk,))
    except Exception as e:
        logger.warning(f"Error deleting from player_game_stats for game {pk}: {e}")

    for side, pfx in [('away','a'), ('home','h')]:
        team_info = box.get(f'{side}', {})
        tid = team_info.get('team',{}).get('id', 0) if team_info else 0
        batters = [b for b in box.get(f'{side}Batters',[])
                   if b.get('personId',0) > 0 and not b.get('substitution',False)]
        for i, b in enumerate(batters[:9], 1):
            pid = b.get('personId', 0)
            if pid > 0 and d is not None:
                try:
                    conn.execute("""INSERT OR IGNORE INTO batter_performances 
                        (game_pk, date, team_id, player_id, player_name, batting_order)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (pk, d, tid, pid, b.get('name',''), str(i*100)))
                except Exception as e:
                    logger.warning(f"Error inserting batter {pid} in game {pk}: {e}")
            for stat in ['avg','ops','slg','obp']:
                val = b.get(stat, None)
                if val is not None:
                    parsed = _safe_float(val, None)
                    if parsed is not None:
                        updates[f'{pfx}_batter_{i}_{stat}'] = parsed

        # Extract individual player stats for player_game_stats
        players = team_info.get('players', {})
        for p_key, p_data in players.items():
            pid_str = p_key.replace('ID', '')
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            batting_stats = p_data.get('stats', {}).get('batting', {})
            if not batting_stats:
                continue
            ab = batting_stats.get('atBats', 0)
            bb = batting_stats.get('baseOnBalls', 0)
            hits = batting_stats.get('hits', 0)
            runs = batting_stats.get('runs', 0)
            if ab == 0 and bb == 0 and hits == 0 and runs == 0:
                continue
            hr = batting_stats.get('homeRuns', 0)
            k = batting_stats.get('strikeOuts', 0)
            doubles = batting_stats.get('doubles', 0)
            triples = batting_stats.get('triples', 0)
            tb = hits + doubles + 2 * triples + 3 * hr
            try:
                conn.execute("""INSERT OR IGNORE INTO player_game_stats 
                    (game_pk, date, player_id, hits, runs, hr, tb, k)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (pk, d, pid, hits, runs, hr, tb, k))
            except Exception as e:
                logger.warning(f"Error inserting player_game_stats for {pid} in game {pk}: {e}")

        if len(batters) >= 9:
            avgs = [_safe_float(b.get('avg'), 0.250) for b in batters[:9]]
            opss = [_safe_float(b.get('ops'), 0.700) for b in batters[:9]]
            obps = [_safe_float(b.get('obp'), 0.320) for b in batters[:9]]
            slgs = [_safe_float(b.get('slg'), 0.400) for b in batters[:9]]
            updates[f'{pfx}_avg'] = sum(avgs)/9
            updates[f'{pfx}_ops'] = sum(opss)/9
            updates[f'{pfx}_obp'] = sum(obps)/9
            updates[f'{pfx}_slg'] = sum(slgs)/9
    if updates:
        # Parameterized SQL to prevent injection (fix 🔴4)
        cols = list(updates.keys())
        vals = list(updates.values())
        sets = ', '.join(f'{c}=?' for c in cols)
        try: conn.execute(f"UPDATE historico_partidos SET {sets} WHERE game_pk=?", vals + [pk])
        except Exception as e: print(f"⚠️ _extract_batters {pk}: {e}")

def _enrich_missing_batters(conn):
    """Retroactively fill batter stats for games missing them. Fix 🔵13: logs progress."""
    try: import statsapi
    except: return
    missing = conn.execute(
        "SELECT game_pk, date FROM historico_partidos WHERE (h_batter_1_ops IS NULL OR h_batter_1_ops=0) AND date >= '2026-01-01' ORDER BY date"
    ).fetchall()
    if not missing:
        return
    total_missing = len(missing)
    batch_size = min(100, total_missing)  # Cap per run
    enriched = 0; failed = 0
    print(f"{C['C']}Enriching {batch_size}/{total_missing} games missing batter data...{C['X']}")
    for idx, (pk, d) in enumerate(missing[:batch_size]):
        try:
            box = statsapi.boxscore_data(pk)
            _extract_batters(conn, box, pk, d)
            enriched += 1
        except Exception as e:
            failed += 1
        if (idx + 1) % 25 == 0:
            conn.commit()
            print(f"  Progress: {idx+1}/{batch_size} (enriched={enriched}, failed={failed})")
    conn.commit()
    remaining = total_missing - batch_size
    print(f"{C['G']}{enriched}/{batch_size} games enriched. {remaining} remaining for next run.{C['X']}")

def load_data(max_date=None):
    conn=sqlite3.connect(DB_PATH)
    q_df = "SELECT * FROM historico_partidos"
    if max_date:
        q_df += f" WHERE date < '{max_date}'"
    q_df += " ORDER BY date"
    df=pd.read_sql(q_df,conn)
    
    q_pp = "SELECT * FROM pitcher_performances"
    if max_date:
        q_pp += f" WHERE date < '{max_date}'"
    q_pp += " ORDER BY date"
    try: pp=pd.read_sql(q_pp,conn)
    except: pp=pd.DataFrame()
    conn.close()
    if not df.empty:
        df['date']=pd.to_datetime(df['date'],format='mixed',errors='coerce')
        df=df.dropna(subset=['date'])
        # Fix #40: include March for spring training, exclude Feb
        df=df[df['date'].dt.month.isin([3,4,5,6,7,8,9,10,11])]
    if not pp.empty:
        pp['date']=pd.to_datetime(pp['date'],format='mixed',errors='coerce')
        pp=pp.dropna(subset=['date'])
    return df,pp

def get_pitcher_db_season_stats(pid, year):
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            """SELECT k, bb, ip, er, hits, batters_faced
               FROM pitcher_performances
               WHERE player_id=? AND role='starter'
               AND date >= ? AND date <= ?""",
            (int(pid), f'{year}-01-01', f'{year}-12-31')
        ).fetchall()
        conn.close()
        if rows:
            k = sum(r[0] or 0 for r in rows)
            bb = sum(r[1] or 0 for r in rows)
            ip = sum(parse_baseball_ip(r[2]) for r in rows)
            er = sum(r[3] or 0 for r in rows)
            ha = sum(r[4] or 0 for r in rows)
            bf_s = sum(r[5] or 0 for r in rows)
            if ip >= 1.0:
                bf = bf_s if (bf_s > 0) else max(int(ip * 4.35) + ha + bb, 1)
                era  = round(er * 9 / ip, 2)
                whip = round((ha + bb) / ip, 2)
                sr   = round(k / bf, 6)
                wr   = round(bb / bf, 6)
                return {'era': era, 'whip': whip,
                        'strikeout_rate': min(sr, 0.42),
                        'walk_rate': min(wr, 0.22),
                        'kbb': round(k / max(bb, 1), 2),
                        'ip': ip, 'source': 'db_hist'}
    except Exception as e:
        logger.debug(f"DB stats failed for pid {pid} year {year}: {e}")
    return None

def get_pitcher_id_by_name_db(name):
    """Fallback ID resolver when StatsAPI schedule doesn't provide IDs.
    Normalizes accents to ensure names like 'Carlos Rodon' match 'Carlos Rodón'.
    """
    if not name or name == 'TBD':
        return 0
    
    def normalize_text(text):
        import unicodedata
        return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    try:
        conn = sqlite3.connect(DB_PATH)
        # Try pitcher_performances first with exact match
        row = conn.execute("SELECT player_id FROM pitcher_performances WHERE player_name=? ORDER BY date DESC LIMIT 1", (name,)).fetchone()
        if row:
            conn.close()
            return int(row[0])
            
        # Try players_master with exact match
        row = conn.execute("SELECT player_id FROM players_master WHERE player_name=?", (name,)).fetchone()
        if row:
            conn.close()
            return int(row[0])
            
        # Fallback: case and accent-insensitive scan
        norm_name = normalize_text(name).lower()
        
        # Scan pitcher_performances
        rows = conn.execute("SELECT DISTINCT player_id, player_name FROM pitcher_performances").fetchall()
        for pid, pname in rows:
            if pname and normalize_text(pname).lower() == norm_name:
                conn.close()
                return int(pid)
                
        # Scan players_master
        rows = conn.execute("SELECT DISTINCT player_id, player_name FROM players_master").fetchall()
        for pid, pname in rows:
            if pname and normalize_text(pname).lower() == norm_name:
                conn.close()
                return int(pid)
                
        conn.close()
        return 0
    except Exception as e:
        logger.debug(f"Failed resolving pitcher ID for {name}: {e}")
        return 0

def get_pitcher_stats(pid, season_year=None, before_date=None):
    """Get pitcher stats for the CURRENT season with Bayesian Blending.
    If current season IP < 30.0, blends with previous season stats if available.
    Supports before_date to prevent data leakage in historical / pre-game simulations.
    """
    if not pid or int(pid) <= 0:
        return None
    if season_year is None:
        season_year = datetime.now().year

    curr = None

    # --- Ruta 1: statsapi live (current season, most accurate) ---
    # Bypass Ruta 1 if before_date is specified to avoid loading live stats that might have today's game
    if before_date is None:
        try:
            import statsapi
            d = statsapi.player_stat_data(personId=int(pid), group="[pitching]", type="season")
            s = d.get('stats', [{}])[0].get('stats', {}) if d else {}
            if s:
                ip_raw = s.get('inningsPitched', 0.0)
                ip = parse_baseball_ip(ip_raw)
                if ip >= 5.0:  # minimum sample for rates to be meaningful
                    k  = int(s.get('strikeOuts', 0) or 0)
                    bb = int(s.get('baseOnBalls', 0) or 0)
                    ha = int(s.get('hits', 0) or 0)
                    er = int(s.get('earnedRuns', 0) or 0)
                    bf = int(s.get('battersFaced', 0) or 0)
                    if bf <= 0: bf = max(int(ip * 4.35) + ha + bb, max(int(k * 1.25), 1))
                    curr = {'era': round(er * 9 / ip, 2), 'whip': round((ha + bb) / ip, 2),
                            'strikeout_rate': min(k / bf, 0.42), 'walk_rate': min(bb / bf, 0.22),
                            'kbb': round(k / max(bb, 1), 2), 'ip': ip, 'source': 'api'}
        except Exception as e:
            logger.debug(f"API stats failed for pid {pid}: {e}")

    # --- Ruta 2: DB local, filtrada a temporada actual, mínimo 5 IP ---
    if curr is None:
        try:
            conn = sqlite3.connect(DB_PATH)
            if before_date:
                rows = conn.execute(
                    """SELECT k, bb, ip, er, hits, batters_faced
                       FROM pitcher_performances
                       WHERE player_id=? AND role='starter'
                       AND date >= ? AND date < ?""",
                    (int(pid), f'{season_year}-01-01', before_date)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT k, bb, ip, er, hits, batters_faced
                       FROM pitcher_performances
                       WHERE player_id=? AND role='starter'
                       AND date >= ?""",
                    (int(pid), f'{season_year}-01-01')
                ).fetchall()
            conn.close()
            if rows:
                k = sum(r[0] or 0 for r in rows)
                bb = sum(r[1] or 0 for r in rows)
                ip = sum(parse_baseball_ip(r[2]) for r in rows)
                er = sum(r[3] or 0 for r in rows)
                ha = sum(r[4] or 0 for r in rows)
                bf_s = sum(r[5] or 0 for r in rows)
                if ip >= 5.0:  # require at minimum 5 real innings to blend
                    bf = bf_s if (bf_s > 0) else max(int(ip * 4.35) + ha + bb, 1)
                    curr = {'era': round(er * 9 / ip, 2), 'whip': round((ha + bb) / ip, 2),
                            'strikeout_rate': min(k / bf, 0.42), 'walk_rate': min(bb / bf, 0.22),
                            'kbb': round(k / max(bb, 1), 2), 'ip': ip, 'source': 'db'}
        except Exception as e:
            logger.debug(f"DB stats failed for pid {pid}: {e}")

    # --- Ruta 3: Blending ---
    curr_ip = curr['ip'] if curr else 0.0
    if curr_ip < 30.0:
        prev = get_pitcher_db_season_stats(pid, season_year - 1)
        if prev:
            if curr:
                # Blend
                w = curr_ip / 30.0
                blended = {'source': f"{curr['source']}+blend"}
                for key in ['era', 'whip', 'strikeout_rate', 'walk_rate', 'kbb']:
                    blended[key] = w * curr[key] + (1 - w) * prev[key]
                blended['ip'] = curr_ip
                return blended
            else:
                # Fallback to previous season stats
                return prev

    return curr


def get_pitcher_savant_stats(pid, date_str):
    """Fetch pitcher Savant statistics (recent starts form & degradation deltas) strictly before date_str."""
    if not pid or int(pid) <= 0:
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("""
            SELECT * FROM savant_pitcher_daily 
            WHERE player_id = ? AND game_date < ? 
            ORDER BY game_date
        """, conn, params=(int(pid), date_str))
        conn.close()
        
        if df.empty:
            return None
            
        df['game_date_dt'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
        gd_dt = pd.to_datetime(date_str, format='mixed', errors='coerce')
        
        # Aggregate daily to start level
        df['ev_prod'] = df['avg_ev_allowed'] * df['bbe']
        df['xwoba_prod'] = df['avg_xwoba_allowed'] * df['bbe']
        df['spin_prod'] = df['avg_release_spin_rate'] * df['bbe']
        df['ext_prod'] = df['avg_release_extension'] * df['bbe']
        
        df_agg = df.groupby(['player_id', 'game_date_dt']).agg(
            bbe=('bbe', 'sum'),
            barrels=('barrels_allowed', 'sum'),
            hard_hits=('hard_hits_allowed', 'sum'),
            ev_prod=('ev_prod', 'sum'),
            xwoba_prod=('xwoba_prod', 'sum'),
            spin_prod=('spin_prod', 'sum'),
            ext_prod=('ext_prod', 'sum')
        ).reset_index().sort_values('game_date_dt')
        
        df_agg['avg_ev'] = df_agg['ev_prod'] / df_agg['bbe'].replace(0, np.nan)
        df_agg['avg_xwoba'] = df_agg['xwoba_prod'] / df_agg['bbe'].replace(0, np.nan)
        df_agg['avg_spin'] = df_agg['spin_prod'] / df_agg['bbe'].replace(0, np.nan)
        df_agg['avg_ext'] = df_agg['ext_prod'] / df_agg['bbe'].replace(0, np.nan)
        
        # L3 starts
        l3 = df_agg.iloc[-3:]
        sum_bbe_l3 = l3['bbe'].sum()
        if sum_bbe_l3 > 0:
            xwoba = float((l3['avg_xwoba'] * l3['bbe']).sum() / sum_bbe_l3)
            barrel = float(l3['barrels'].sum() / sum_bbe_l3)
            hardhit = float(l3['hard_hits'].sum() / sum_bbe_l3)
            ev = float((l3['avg_ev'] * l3['bbe']).sum() / sum_bbe_l3)
        else:
            xwoba = float(l3['avg_xwoba'].mean())
            barrel = 0.0826
            hardhit = 0.4024
            ev = float(l3['avg_ev'].mean())
            
        spin = float(l3['avg_spin'].mean())
        ext = float(l3['avg_ext'].mean())
        
        # L2 starts
        l2 = df_agg.iloc[-2:]
        spin_l2 = float(l2['avg_spin'].mean())
        ext_l2 = float(l2['avg_ext'].mean())
        sum_bbe_l2 = l2['bbe'].sum()
        if sum_bbe_l2 > 0:
            ev_l2 = float((l2['avg_ev'] * l2['bbe']).sum() / sum_bbe_l2)
        else:
            ev_l2 = float(l2['avg_ev'].mean())
            
        # Season baseline
        df_year = df_agg[df_agg['game_date_dt'].dt.year == gd_dt.year]
        if df_year.empty:
            df_year = df_agg
            
        spin_season = float(df_year['avg_spin'].mean())
        ext_season = float(df_year['avg_ext'].mean())
        sum_bbe_season = df_year['bbe'].sum()
        if sum_bbe_season > 0:
            ev_season = float((df_year['avg_ev'] * df_year['bbe']).sum() / sum_bbe_season)
        else:
            ev_season = float(df_year['avg_ev'].mean())
            
        spin_delta = spin_l2 - spin_season if (not pd.isna(spin_l2) and not pd.isna(spin_season)) else 0.0
        ext_delta = ext_l2 - ext_season if (not pd.isna(ext_l2) and not pd.isna(ext_season)) else 0.0
        ev_delta = ev_l2 - ev_season if (not pd.isna(ev_l2) and not pd.isna(ev_season)) else 0.0
        
        return {
            'xwoba': xwoba, 'barrel_rate': barrel, 'hardhit_rate': hardhit,
            'ev': ev, 'spin': spin, 'extension': ext,
            'spin_delta': spin_delta, 'extension_delta': ext_delta, 'ev_delta': ev_delta
        }
    except Exception as e:
        logger.warning(f"Error fetching Savant stats for pitcher {pid}: {e}")
        return None

def get_batter_savant_stats(pid, date_str, opp_hand='R'):
    """Fetch batter Savant statistics (platoon splits + recent form) strictly before date_str."""
    if not pid or int(pid) <= 0:
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("""
            SELECT * FROM savant_batter_daily 
            WHERE player_id = ? AND game_date < ? 
            ORDER BY game_date
        """, conn, params=(int(pid), date_str))
        conn.close()
        
        if df.empty:
            return None
            
        df['game_date_dt'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
        
        overall_l15 = df.iloc[-15:]
        df_split = df[df['p_throws'] == opp_hand]
        split_l15 = df_split.iloc[-15:]
        
        def safe_weighted_mean(df_sub, val_col, wt_col, fallback):
            if df_sub.empty: return fallback
            w = df_sub[wt_col].fillna(0)
            v = df_sub[val_col].fillna(fallback)
            sum_w = w.sum()
            if sum_w > 0:
                return float((v * w).sum() / sum_w)
            return float(v.mean()) if not v.empty else fallback

        def safe_ratio(df_sub, num_col, den_col, fallback):
            if df_sub.empty: return fallback
            num = df_sub[num_col].fillna(0).sum()
            den = df_sub[den_col].fillna(0).sum()
            if den > 0:
                return float(num / den)
            return fallback

        metrics = {}
        for key, df_sub in [('split', split_l15), ('overall', overall_l15)]:
            metrics[f'{key}_xwoba'] = safe_weighted_mean(df_sub, 'avg_xwoba', 'bbe', 0.320)
            metrics[f'{key}_barrel'] = safe_ratio(df_sub, 'barrels', 'bbe', 0.080)
            metrics[f'{key}_hardhit'] = safe_ratio(df_sub, 'hard_hits', 'bbe', 0.400)
            metrics[f'{key}_ev'] = safe_weighted_mean(df_sub, 'avg_ev', 'bbe', 89.0)
            metrics[f'{key}_bat_speed'] = safe_weighted_mean(df_sub, 'avg_bat_speed', 'bbe', 71.5)
            metrics[f'{key}_swing_length'] = safe_weighted_mean(df_sub, 'avg_swing_length', 'bbe', 7.3)
            metrics[f'{key}_sweetspot'] = safe_ratio(df_sub, 'sweet_spot_count', 'bbe', 0.33)
            
            z_rate = safe_ratio(df_sub, 'z_swings', 'z_pitches', 0.65)
            o_rate = safe_ratio(df_sub, 'o_swings', 'o_pitches', 0.30)
            metrics[f'{key}_discipline'] = z_rate / (o_rate + 0.001)
            metrics[f'{key}_efficiency'] = metrics[f'{key}_bat_speed'] / max(metrics[f'{key}_swing_length'], 0.1)
            
        bbe_split = split_l15['bbe'].sum() if not split_l15.empty else 0
        weight = min(1.0, float(bbe_split / 10.0))
        
        defaults = {
            'xwoba': 0.320, 'barrel_rate': 0.080, 'hardhit_rate': 0.400, 'ev': 89.0,
            'bat_speed': 71.5, 'swing_length': 7.3, 'sweetspot': 0.33,
            'discipline': 1.5, 'efficiency': 9.8
        }
        
        blended = {}
        for m in ['xwoba', 'barrel', 'hardhit', 'ev', 'bat_speed', 'swing_length', 'sweetspot', 'discipline', 'efficiency']:
            key_m = 'barrel_rate' if m == 'barrel' else ('hardhit_rate' if m == 'hardhit' else m)
            s_val = metrics[f'split_{m}']
            o_val = metrics[f'overall_{m}']
            val = weight * s_val + (1.0 - weight) * o_val
            if pd.isna(val): val = defaults[key_m]
            blended[key_m] = val
            
        return blended
    except Exception as e:
        logger.warning(f"Error fetching Savant stats for batter {pid}: {e}")
        return None

def get_bullpen_savant_stats(team_id, date_str):
    if not team_id: return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("""
            SELECT DISTINCT player_id FROM pitcher_performances 
            WHERE team_id = ? AND role = 'reliever' AND date < ? AND date >= date(?, '-14 days')
        """, conn, params=(team_id, date_str, date_str))
        conn.close()
        
        if df.empty: return None
        
        xwoba_list = []
        barrel_list = []
        
        for pid in df['player_id']:
            stats = get_pitcher_savant_stats(pid, date_str)
            if stats:
                if stats['xwoba'] is not None: xwoba_list.append(stats['xwoba'])
                if stats['barrel_rate'] is not None: barrel_list.append(stats['barrel_rate'])
                
        return {
            'xwoba': np.mean(xwoba_list) if xwoba_list else None,
            'barrel_rate': np.mean(barrel_list) if barrel_list else None
        }
    except Exception as e:
        logger.warning(f"Error fetching bullpen savant stats for team {team_id}: {e}")
        return None


def ensure_models():
    if USE_V3_ENSEMBLE:
        if not v3_model_exists():
            print(f"{C['Y']}V3 model not found. Training...{C['X']}")
            import subprocess
            subprocess.run(["python3", "models/train_ensemble_v3.py"], check=True)
    else:
        if not _legacy_models_exist() or needs_retrain():
            print(f"{C['Y']}Training models...{C['X']}")
            df,pp=load_data()
            df,feats,target=build_all_features(df,pp)
            train_multiwindow(df,feats,target)

def backtest(beta=False):
    print("="*90+"\n OMEGA v3 BACKTEST\n"+"="*90)
    df,pp=load_data()
    df = df.reset_index(drop=True)
    # KNOWN ISSUE: build_all_features processes ALL games (including test period)
    # before train/test split. ELO, H2H, streaks leak future info into training.
    # This makes backtest results overly optimistic; does NOT affect live predictions.
    # To fix: compute features incrementally per-game in chronological order.
    df,feats,target=build_all_features(df,pp)
    elo = None; starters = None
    if beta:
        try:
            from models.elo import PlayerELO
            elo = PlayerELO()
            starters = pp[pp['role']=='starter'] if pp is not None and not pp.empty else None
            print(f"{C['P']}🧬 BACKTEST WITH BETA ELO ADJUSTMENTS{C['X']}")
        except Exception as e: print(f"BETA err: {e}")
    # Features are built ONCE — reuse across all splits
    splits=[(2023,2024),(2024,2025),(2025,2026)]
    results=[]; _audit_rows=[]
    for ty,ey in splits:
        trm=(df['date'].dt.year<=ty)&(df['date'].dt.year>=2022)
        tem=df['date'].dt.year==ey
        dtr,dte=df[trm],df[tem]
        if len(dtr)<100 or len(dte)<10: continue
        tgt_tr = target[trm.values]
        tgt_te = target[tem.values]
        # Train with pre-computed features (no rebuild needed)
        models,scalers,cal=train_multiwindow(dtr,feats,tgt_tr,save=False)
        # BATCH prediction: vectorized instead of row-by-row
        from models.game import WEIGHTS
        X_te = dte[feats].fillna(0).values
        if 'all' not in scalers: continue
        X_te_s = scalers['all'].transform(X_te)
        # Get weighted ensemble probabilities for all test games at once
        probs = np.zeros(len(X_te))
        indiv = []
        for mname, weight in WEIGHTS.items():
            key = f'all_{mname}'
            if key not in models: continue
            p = models[key].predict_proba(X_te_s)[:,1]
            probs += p * weight
            indiv.append(p)
        probs /= sum(WEIGHTS.values())
        # GOD MODE: all models agree strongly
        if len(indiv) >= 5:
            indiv_arr = np.array(indiv)
            god_mask = np.all(indiv_arr > 0.65, axis=0) | np.all(indiv_arr < 0.35, axis=0)
        else:
            god_mask = np.zeros(len(X_te), dtype=bool)
        # Window consensus
        cons_arr = np.ones(len(X_te), dtype=int)  # 'all' always agrees with itself
        for wname in ['w50','w25','w10']:
            if wname not in scalers: continue
            X_w = scalers[wname].transform(X_te)
            wp = np.zeros(len(X_te))
            for mname, weight in WEIGHTS.items():
                key = f'{wname}_{mname}'
                if key not in models: continue
                wp += models[key].predict_proba(X_w)[:,1] * weight
            wp /= sum(WEIGHTS.values())
            cons_arr += ((wp >= 0.5) == (probs >= 0.5)).astype(int)
        # Build results
        picks = (probs >= 0.5).astype(int)
        conf_raw = np.maximum(probs, 1-probs) * 100
        for i in range(len(tgt_te)):
            cr = round(conf_raw[i], 1)
            # Apply calibration lookup manually for batch backtest (fix 🐛 NEW BUG)
            bucket = f"{int(cr//5)*5}_{int(cr//5)*5+5}"
            cc = cal.get(bucket, cr) if cal else cr
            
            # Apply Form & Volatility Penalties (to match predict_live)
            row = dte.iloc[i]
            pick_is_home = (picks[i] == 1)
            p_streak = row.get('h_streak', 0) if pick_is_home else row.get('a_streak', 0)
            opp_streak = row.get('a_streak', 0) if pick_is_home else row.get('h_streak', 0)
            p_std = row.get('h_starter_era_std', 3.5) if pick_is_home else row.get('a_starter_era_std', 3.5)
            
            is_hot = p_streak >= 4 and opp_streak <= 2
            is_cold = p_streak <= 2 and opp_streak >= 4
            is_high_vol = p_std > 6.0
            
            if is_high_vol:
                cc -= 3.5
                cc = min(cc, 74.9)
            if is_cold and cc >= 78 and cc < 80:
                cc -= 4.6
                
            beta_tag = ""
            if beta and elo and starters is not None:
                row = dte.iloc[i]
                gpk = row.get('game_pk')
                ht, at = row['home_team'], row['away_team']
                htid, atid = TEAM_IDS.get(ht), TEAM_IDS.get(at)
                st = starters[starters['game_pk']==gpk]
                hm = st[st['team_id']==htid]
                am = st[st['team_id']==atid]
                if not hm.empty and not am.empty:
                    hpid = int(hm.iloc[0]['player_id'])
                    apid = int(am.iloc[0]['player_id'])
                    helo = elo.get_historical_pitcher_elo(hpid, gpk)
                    aelo = elo.get_historical_pitcher_elo(apid, gpk)
                    sup = (helo - aelo) / 100.0
                    sup = max(-1.0, min(1.0, sup))
                    delta = sup * 0.08
                    pick_side = ht if probs[i] >= 0.5 else at
                    if pick_side == at: delta = -delta  # If pick is away, higher home elo hurts pick
                    cc = min(99.0, max(50.0, cc + delta * 100))
                    arr = "▲" if delta > 0 else "▼"
                    beta_tag = f"🧬{arr}{abs(delta*100):.1f}%"
            results.append({'year':ey,'prob':probs[i],'conf_raw':cr,
                          'conf_cal':round(cc,1),'god':bool(god_mask[i]),
                          'consensus':int(cons_arr[i]),'pick':int(picks[i]),
                          'real':int(tgt_te[i]),'hit':int(picks[i]==tgt_te[i])})
    # ── AUDIT CSV accumulation ──
    for i in range(len(dte)):
        prob = float(probs[i])
        actual = int(tgt_te[i])
        pick = 1 if prob >= 0.5 else 0
        conf = round(max(prob, 1 - prob) * 100, 1)
        bucket = f"{int(conf // 5) * 5}-{int(conf // 5) * 5 + 5}"
        row = dte.iloc[i]
        _audit_rows.append({
            'fecha': str(row['date'])[:10],
            'home': row['home_team'],
            'away': row['away_team'],
            'prob_home': round(prob, 4),
            'confianza': conf,
            'bucket': bucket,
            'pick_home': pick,
            'resultado_home': actual,
            'acerto': int(pick == actual),
            'year': str(row['date'])[:4]
        })
    if not results: return
    # ── Save audit CSV ──
    if _audit_rows:
        audit_path = os.path.join(BASE_DIR, 'backtest_audit.csv')
        with open(audit_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_audit_rows[0].keys())
            writer.writeheader()
            writer.writerows(_audit_rows)
        print(f"\nAudit CSV guardado: {audit_path}")
        print("\n=== CALIBRACIÓN REAL POR BUCKET ===")
        _buckets = {}
        for r in _audit_rows:
            b = r['bucket']
            _buckets.setdefault(b, {'n': 0, 'hits': 0})
            _buckets[b]['n'] += 1
            _buckets[b]['hits'] += r['acerto']
        for b in sorted(_buckets.keys()):
            d = _buckets[b]
            if d['n'] >= 10:
                acc = d['hits'] / d['n'] * 100
                print(f"  {b}%: {acc:.1f}% accuracy ({d['n']} juegos)")
    R=pd.DataFrame(results)
    acc=R['hit'].mean(); profit=((R['hit']*1.91)-1).sum()
    print(f"\n  Accuracy: {acc:.2%} ({R['hit'].sum()}/{len(R)})")
    print(f"  Profit:   {profit:+.2f}u")
    conf_col = 'conf_cal'
    
    print(f"\n  RENDIMIENTO POR TIERS (con filtros de racha y volatilidad):")
    def get_tier(row):
        if row['god'] and row[conf_col] >= 80: return "💎 GOD"
        elif row[conf_col] >= 78: return "🎯 SNIPER"
        elif row[conf_col] >= 65: return "📦 VOLUMEN"
        else: return "📊 STANDARD"
        
    R['tier'] = R.apply(get_tier, axis=1)
    
    for t in ["💎 GOD", "🎯 SNIPER", "📦 VOLUMEN", "📊 STANDARD"]:
        subset = R[R['tier'] == t]
        if len(subset) > 0:
            print(f"  {t}: {subset['hit'].mean():.2%} ({subset['hit'].sum()}/{len(subset)})")
    # By consensus
    for c in [4,3]:
        sc=R[R['consensus']>=c]
        if len(sc)>=5: print(f"  Consensus≥{c}: {sc['hit'].mean():.2%} ({sc['hit'].sum()}/{len(sc)})")
    # By year
    for yr in sorted(R['year'].unique()):
        y=R[R['year']==yr]; print(f"  {yr}: {y['hit'].mean():.2%} ({y['hit'].sum()}/{len(y)})")
    # Calibration check
    print(f"\n  CALIBRATION CHECK:")
    for lo in range(50,95,5):
        hi=lo+5
        s=R[(R[conf_col]>=lo)&(R[conf_col]<hi)]
        if len(s)>=10: print(f"    {lo}-{hi}%: actual={s['hit'].mean():.1%} ({len(s)} games)")
    print("="*90)

def guess_team_lineup(team_name, date_str):
    """Lineup Guesser BETA (Sum of Weighted Recent + Markov Predictor)."""
    conn = sqlite3.connect(DB_PATH)
    query_tid = "SELECT team_id FROM historico_partidos hp JOIN batter_performances bp ON hp.game_pk = bp.game_pk WHERE (hp.home_team=? OR hp.away_team=?) AND hp.date < ? LIMIT 1"
    row = conn.execute(query_tid, (team_name, team_name, date_str)).fetchone()
    conn.close()
    if not row:
        return []
    tid = row[0]
    
    try:
        from sync import LineupPredictor
        lp = LineupPredictor()
        # This will return the Top 9 weighted array assigned to optimal Markov slots
        res = lp.predict_lineup(tid, date_str)
        return res if res else []
    except Exception as e:
        return []

def find_next_game_date(start_date_str=None, max_days=7):
    """Find the next date with scheduled MLB games that haven't started yet."""
    try: import statsapi
    except: return None
    if start_date_str:
        base = datetime.strptime(start_date_str, '%Y-%m-%d')
    else:
        base = datetime.now()
    for offset in range(max_days + 1):
        check_date = base + timedelta(days=offset)
        date_str = check_date.strftime('%Y-%m-%d')
        try:
            games = statsapi.schedule(date=date_str)
            if not games:
                continue
            # Check if any game hasn't started yet
            upcoming_statuses = {'Pre-Game', 'Scheduled', 'Warmup'}
            has_upcoming = any(g.get('status') in upcoming_statuses for g in games)
            if has_upcoming:
                return date_str
        except Exception:
            continue
    # Fallback: return any date with games
    for offset in range(max_days + 1):
        check_date = base + timedelta(days=offset)
        date_str = check_date.strftime('%Y-%m-%d')
        try:
            games = statsapi.schedule(date=date_str)
            if games:
                return date_str
        except Exception:
            continue
    return None

def auto_sync_and_retrain(skip_sync=False):
    """Auto-sync yesterday's games and retrain if week has passed."""
    try:
        if not skip_sync:
            print(f"{C['C']}🔄 Auto-sincronizando ayer...{C['X']}")
            sync_data(1)
            print(f"{C['C']}🔄 Sincronizando Statcast...{C['X']}")
            sync_savant_daily()
        if needs_retrain():
            print(f"{C['Y']}🧠 Reentrenando modelos (semana completa)...{C['X']}")
            df, pp = load_data()
            df, f, t = build_all_features(df, pp)
            train_multiwindow(df, f, t)
            print(f"{C['G']}✅ Modelos reentrenados exitosamente{C['X']}")
    except Exception as e:
        logger.warning(f"Auto sync/retrain failed: {e}")

def predict_live(date_str=None,beta=False, guess_lineups=False, use_filters=True, skip_sync=False, skip_train=False):
    try: import statsapi
    except: print("pip install statsapi"); return
    # Auto-sync and retrain first
    if not skip_train:
        auto_sync_and_retrain(skip_sync=skip_sync)
    elif not skip_sync:
        sync_data(days_back=3)
    # If date_str is explicitly provided, respect it. Otherwise, find next upcoming.
    if not date_str:
        next_date = find_next_game_date(None)
        date_str = next_date if next_date else datetime.now().strftime('%Y-%m-%d')
    if not skip_train:
        ensure_models()
    if USE_V3_ENSEMBLE:
        models,scalers,feats_legacy,cal=None,None,None,None
    else:
        models,scalers,feats_legacy,cal=load_all_models()
    # Create PlayerELO once
    elo_sys=None
    beta_analyzer=None
    try:
        from models.elo import PlayerELO, LineupAnalyzer
        elo_sys=PlayerELO()
        if beta:
            beta_analyzer=LineupAnalyzer(elo_sys)
            print(f"{C['P']}🧬 BETA ON{C['X']}")
    except Exception as e:
        print(f"PlayerELO/BETA initialization error: {e}")
    # Historical team stats
    conn=sqlite3.connect(DB_PATH)
    try:
        dfh=pd.read_sql("SELECT * FROM historico_partidos ORDER BY date",conn)
        dfh['date']=pd.to_datetime(dfh['date'],format='mixed',errors='coerce')
    except Exception as e:
        logger.error(f"Error loading historico_partidos: {e}")
        dfh = pd.DataFrame()
    try:
        pph=pd.read_sql("SELECT * FROM pitcher_performances ORDER BY date",conn)
        pph['date']=pd.to_datetime(pph['date'],format='mixed',errors='coerce')
    except Exception as e:
        logger.warning(f"Error loading pitcher_performances: {e}")
        pph = pd.DataFrame()
    conn.close()

    # Filter out games on or after date_str to avoid data leakage
    if not dfh.empty:
        dfh = dfh[dfh['date'] < pd.to_datetime(date_str)]
    if not pph.empty:
        pph = pph[pph['date'] < pd.to_datetime(date_str)]

    # Build ts and elo_map in single pass (fix #42: avoid 2x iterrows)
    ts={}; elo_map={}; h2h_wins={}
    for r in dfh.itertuples(index=False):
        ht,at=r.home_team,r.away_team
        for t in[ht,at]:
            if t not in ts: ts[t]={'ops':[],'wins':[],'avg':[],'obp':[],'slg':[],'runs_for':[],'runs_ag':[],'home_wins':[],'away_wins':[]}
        hr,ar_=r.h_runs_total,r.a_runs_total
        ts[ht]['runs_for'].append(hr); ts[ht]['runs_ag'].append(ar_)
        ts[at]['runs_for'].append(ar_); ts[at]['runs_ag'].append(hr)
        won_h=(hr>ar_)
        ts[ht]['home_wins'].append(1 if won_h else 0)
        ts[at]['away_wins'].append(1 if not won_h else 0)
        if won_h: h2h_wins[(ht,at)]=h2h_wins.get((ht,at),0)+1
        else: h2h_wins[(at,ht)]=h2h_wins.get((at,ht),0)+1
        for team,s in[(ht,'h'),(at,'a')]:
            for stat in['ops','avg','obp','slg']:
                v=getattr(r, f'{s}_{stat}', None)
                # Fix #41: accept 0.0 as valid stat
                if v is not None and not pd.isna(v): ts[team][stat].append(float(v))
            won=(s=='h' and hr>ar_) or (s=='a' and ar_>hr)
            ts[team]['wins'].append(1 if won else 0)
        # ELO in same pass
        eh,ea=elo_map.get(ht,1500),elo_map.get(at,1500)
        w=1 if hr>ar_ else 0
        e=1/(1+10**((ea-eh)/400))
        elo_map[ht]=eh+20*(w-e); elo_map[at]=ea+20*((1-w)-(1-e))
    starter_era_hist = {}
    if pph is not None and not pph.empty:
        # Only use real era_game values recorded in DB — skip NULL/missing entries
        starters_real = pph[(pph['role']=='starter') & pph['era_game'].notna()].sort_values('date')
        for _, sr in starters_real.iterrows():
            pid = int(sr['player_id'])
            starter_era_hist.setdefault(pid, []).append(float(sr['era_game']))
    today_str = datetime.now().strftime('%Y-%m-%d')
    if date_str > today_str:
        print(f"\n📡 SIGUIENTE JORNADA: {date_str}...")
    elif date_str < today_str:
        print(f"\n📡 JORNADA PASADA: {date_str}...")
    else:
        print(f"\n📡 HOY: {date_str}...")
    try:
        games=statsapi.schedule(date=date_str)
    except Exception as e:
        logger.warning(f"Schedule API failed: {e}")
        if skip_sync:
            print(f"{C['D']}⚠️ API bloqueada. Usando schedule de DB...{C['X']}")
            # Try to get games from DB
            conn_sched = sqlite3.connect(DB_PATH)
            db_games = pd.read_sql(
                "SELECT game_pk, home_team, away_team, date FROM historico_partidos WHERE date = ?",
                conn_sched, params=(date_str,))
            conn_sched.close()
            if db_games.empty:
                print(f"No games found for {date_str}"); return
            games = []
            for _, row in db_games.iterrows():
                games.append({
                    'game_id': row['game_pk'],
                    'home_name': row['home_team'],
                    'away_name': row['away_team'],
                    'status': 'Final',
                    'home_probable_pitcher': 'TBD',
                    'away_probable_pitcher': 'TBD',
                    'home_probable_pitcher_id': 0,
                    'away_probable_pitcher_id': 0,
                })
        else:
            raise
    valid={'Pre-Game','Scheduled','Warmup','In Progress','Final','Completed','Postponed','Cancelled'}
    games=[g for g in games if g.get('status') in valid]
    if not games: print("No games"); return
    results=[]
    game_data=[]
    today_dicts=[]
    for g in games:
        gid=g['game_id']; hn,an=g['home_name'],g['away_name']
        hp_id=g.get('home_probable_pitcher_id',0)
        ap_id=g.get('away_probable_pitcher_id',0)
        hp=g.get('home_probable_pitcher','TBD')
        ap=g.get('away_probable_pitcher','TBD')
        if hp_id==0: hp_id=get_pitcher_id_by_name_db(hp)
        if ap_id==0: ap_id=get_pitcher_id_by_name_db(ap)
        confirmed=(hp!='TBD' and ap!='TBD')
        if not confirmed: print(f"  {C['D']}⏭️ {an} @ {hn}: TBD{C['X']}"); continue
        lineup_confirmed=False; hb,ab=[],[]
        # Fix #46: only fetch boxscore if game status suggests lineups might be posted
        if g.get('status') in ('Pre-Game','Warmup','In Progress'):
            try:
                box=statsapi.boxscore_data(gid)
                hb_temp=[b for b in box.get('homeBatters',[]) if b.get('personId',0)>0 and not b.get('substitution',False)]
                ab_temp=[b for b in box.get('awayBatters',[]) if b.get('personId',0)>0 and not b.get('substitution',False)]
                if len(hb_temp)>=9 and len(ab_temp)>=9:
                    lineup_confirmed=True
                    hb=hb_temp[:9]
                    ab=ab_temp[:9]
            except Exception as e: logger.debug(f"Boxscore fetch failed for {gid}: {e}")
        if not lineup_confirmed:
            try:
                lp=get_lineup_predictor()
                htid=TEAM_IDS.get(hn); atid=TEAM_IDS.get(an)
                pred_h=lp.predict_lineup(htid, date_str)
                pred_a=lp.predict_lineup(atid, date_str)
                if pred_h and len(pred_h)>=9: hb=pred_h
                if pred_a and len(pred_a)>=9: ab=pred_a
            except Exception as e:
                logger.debug(f"Lineup predictor failed: {e}")

        if lineup_confirmed:
            scan="✅ CONFIRMED"
        elif hb and ab:
            scan="🔮 GUESS-SCAN"
        elif guess_lineups:
            hb_guess = guess_team_lineup(hn, date_str)
            ab_guess = guess_team_lineup(an, date_str)
            if len(hb_guess) == 9 and len(ab_guess) == 9:
                hb = hb_guess
                ab = ab_guess
                scan = "🔮 GUESS-SCAN"
            else:
                scan = "🅿️ PRE-SCAN"
        else:
            scan="🅿️ PRE-SCAN"
            
        sc=C['G'] if (lineup_confirmed and scan != "🔮 GUESS-SCAN") else C['C']
        
        hs  = get_pitcher_stats(hp_id, before_date=date_str)
        ast = get_pitcher_stats(ap_id, before_date=date_str)
        # If either pitcher has insufficient real data, mark game as LOW CONFIDENCE and skip SNIPER+ tiers
        pitcher_data_complete = (hs is not None and ast is not None)
        if not pitcher_data_complete:
            # Use league-median values but flag that confidence must be capped at 64%
            _league = {'era':4.20,'whip':1.32,'strikeout_rate':0.220,'walk_rate':0.085,'kbb':2.60,'ip':0,'source':'insufficient'}
            if hs  is None: hs  = _league
            if ast is None: ast = _league
        # Build feature dict with ALL features
        d={}
        # Identity columns required by build_all_features (groupby, adjust_pitcher_stats_blended)
        d['home_team'] = hn
        d['away_team'] = an
        d['h_starter'] = hp
        d['a_starter'] = ap
        # Pitcher stats
        d['h_starter_era']=hs['era']; d['a_starter_era']=ast['era']
        d['h_starter_whip']=hs['whip']; d['a_starter_whip']=ast['whip']
        d['h_starter_strikeout_rate']=hs['strikeout_rate']; d['a_starter_strikeout_rate']=ast['strikeout_rate']
        d['h_starter_walk_rate']=hs['walk_rate']; d['a_starter_walk_rate']=ast['walk_rate']
        # Savant stats
        hs_sav = get_pitcher_savant_stats(hp_id, date_str)
        ast_sav = get_pitcher_savant_stats(ap_id, date_str)
        d['h_starter_xwoba'] = hs_sav['xwoba'] if hs_sav and hs_sav['xwoba'] is not None else 0.3192
        d['a_starter_xwoba'] = ast_sav['xwoba'] if ast_sav and ast_sav['xwoba'] is not None else 0.3191
        d['h_starter_barrel'] = hs_sav['barrel_rate'] if hs_sav and hs_sav['barrel_rate'] is not None else 0.0826
        d['a_starter_barrel'] = ast_sav['barrel_rate'] if ast_sav and ast_sav['barrel_rate'] is not None else 0.0826
        d['h_starter_hardhit'] = hs_sav['hardhit_rate'] if hs_sav and hs_sav['hardhit_rate'] is not None else 0.4024
        d['a_starter_hardhit'] = ast_sav['hardhit_rate'] if ast_sav and ast_sav['hardhit_rate'] is not None else 0.4025
        d['h_starter_ev'] = hs_sav['ev'] if hs_sav and hs_sav['ev'] is not None else 88.5
        d['a_starter_ev'] = ast_sav['ev'] if ast_sav and ast_sav['ev'] is not None else 88.5
        d['h_starter_spin'] = hs_sav['spin'] if hs_sav and hs_sav['spin'] is not None else 2200.0
        d['a_starter_spin'] = ast_sav['spin'] if ast_sav and ast_sav['spin'] is not None else 2200.0
        d['h_starter_extension'] = hs_sav['extension'] if hs_sav and hs_sav['extension'] is not None else 6.2
        d['a_starter_extension'] = ast_sav['extension'] if ast_sav and ast_sav['extension'] is not None else 6.2
        
        # New Starter Deltas
        d['h_starter_spin_delta'] = hs_sav['spin_delta'] if hs_sav and hs_sav['spin_delta'] is not None else 0.0
        d['a_starter_spin_delta'] = ast_sav['spin_delta'] if ast_sav and ast_sav['spin_delta'] is not None else 0.0
        d['h_starter_extension_delta'] = hs_sav['extension_delta'] if hs_sav and hs_sav['extension_delta'] is not None else 0.0
        d['a_starter_extension_delta'] = ast_sav['extension_delta'] if ast_sav and ast_sav['extension_delta'] is not None else 0.0
        d['h_starter_ev_delta'] = hs_sav['ev_delta'] if hs_sav and hs_sav['ev_delta'] is not None else 0.0
        d['a_starter_ev_delta'] = ast_sav['ev_delta'] if ast_sav and ast_sav['ev_delta'] is not None else 0.0

        d['h_starter_true_risk'] = (1 - d['h_starter_strikeout_rate']) * d['h_starter_barrel']
        d['a_starter_true_risk'] = (1 - d['a_starter_strikeout_rate']) * d['a_starter_barrel']

        # Look up physical pitcher hands from player_hands cache
        conn = sqlite3.connect(DB_PATH)
        hp_hand_row = conn.execute("SELECT pitch_hand FROM player_hands WHERE player_id = ?", (hp_id,)).fetchone() if hp_id else None
        ap_hand_row = conn.execute("SELECT pitch_hand FROM player_hands WHERE player_id = ?", (ap_id,)).fetchone() if ap_id else None
        hp_hand = hp_hand_row[0] if hp_hand_row else 'R'
        ap_hand = ap_hand_row[0] if ap_hand_row else 'R'

        # Bullpen Fatigue and Savant
        gd = pd.to_datetime(date_str)
        for side, tid, opposing_hand, bp_fat_48_col, bp_fat_72_col, bp_xw_col, bp_bar_col in [
            ('h', TEAM_IDS.get(hn), ap_hand, 'h_bp_fatigue_recent', 'h_bp_fatigue_recent_72h', 'h_bullpen_xwoba', 'h_bullpen_barrel'),
            ('a', TEAM_IDS.get(an), hp_hand, 'a_bp_fatigue_recent', 'a_bp_fatigue_recent_72h', 'a_bullpen_xwoba', 'a_bullpen_barrel')
        ]:
            fat_48, fat_72, b_xw, b_bar = 0.0, 0.0, 0.319, 0.08
            if tid and not pph.empty:
                team_rels = pph[(pph['team_id'] == tid) & (pph['role'] == 'reliever')]
                if not team_rels.empty:
                    team_rels_30d = team_rels[(team_rels['date'] < gd) & (team_rels['date'] >= gd - timedelta(days=30))]
                    if not team_rels_30d.empty:
                        rids = team_rels_30d['player_id'].unique()
                        rel_stats = []
                        for rid in rids:
                            rxw, rbar = 0.319, 0.08
                            r_df = conn.execute("""
                                SELECT avg_xwoba_allowed, barrels_allowed, bbe FROM savant_pitcher_daily
                                WHERE player_id = ? AND game_date < ?
                                ORDER BY game_date DESC LIMIT 5
                            """, (int(rid), date_str)).fetchall()
                            if r_df:
                                sum_bbe = sum(r[2] or 0 for r in r_df)
                                if sum_bbe > 0:
                                    rxw = sum((r[0] or 0.319) * (r[2] or 0) for r in r_df) / sum_bbe
                                    rbar = sum(r[1] or 0 for r in r_df) / sum_bbe
                            rel_stats.append((rid, rxw, rbar))
                        rel_stats.sort(key=lambda x: x[1])
                        top_2_rids = [x[0] for x in rel_stats[:2]]
                        
                        recent_48h = team_rels_30d[team_rels_30d['player_id'].isin(top_2_rids) & (team_rels_30d['date'] >= gd - timedelta(days=2))]
                        recent_72h = team_rels_30d[team_rels_30d['player_id'].isin(top_2_rids) & (team_rels_30d['date'] >= gd - timedelta(days=3))]
                        fat_48 = float(recent_48h['pitches'].sum())
                        fat_72 = float(recent_72h['pitches'].sum())
                        
                        team_rels_14d = team_rels_30d[team_rels_30d['date'] >= gd - timedelta(days=14)]
                        rids_14d = team_rels_14d['player_id'].unique()
                        if len(rids_14d) > 0:
                            xw_list = [x[1] for x in rel_stats if x[0] in rids_14d]
                            bar_list = [x[2] for x in rel_stats if x[0] in rids_14d]
                            b_xw = float(np.mean(xw_list)) if xw_list else 0.319
                            b_bar = float(np.mean(bar_list)) if bar_list else 0.08
            d[bp_fat_48_col] = fat_48
            d[bp_fat_72_col] = fat_72
            d[bp_xw_col] = b_xw
            d[bp_bar_col] = b_bar
        conn.close()

        # Team batting stats — compute rolling 10-game team averages as fallback/pre-scan
        team_rolling = {}
        for pfx,tn in[('h',hn),('a',an)]:
            team_rolling[pfx] = {}
            for stat in['ops','avg','obp','slg']:
                vals=ts.get(tn,{}).get(stat,[])
                if len(vals) >= 10:
                    team_rolling[pfx][stat]=sum(vals[-10:])/10
                elif len(vals) >= 1:
                    team_rolling[pfx][stat]=sum(vals)/len(vals)
                else:
                    team_rolling[pfx][stat]=None
            
            # Fill None with DB medians/defaults
            if team_rolling[pfx]['ops'] is None:
                all_ops_vals = [v for t_vals in [ts.get(t,{}).get('ops',[]) for t in ts] for v in t_vals]
                team_rolling[pfx]['ops'] = sum(all_ops_vals)/len(all_ops_vals) if all_ops_vals else 0.700
            if team_rolling[pfx]['avg'] is None:
                all_avg_vals = [v for t_vals in [ts.get(t,{}).get('avg',[]) for t in ts] for v in t_vals]
                team_rolling[pfx]['avg'] = sum(all_avg_vals)/len(all_avg_vals) if all_avg_vals else 0.250
            if team_rolling[pfx]['obp'] is None:
                all_obp_vals = [v for t_vals in [ts.get(t,{}).get('obp',[]) for t in ts] for v in t_vals]
                team_rolling[pfx]['obp'] = sum(all_obp_vals)/len(all_obp_vals) if all_obp_vals else 0.320
            if team_rolling[pfx]['slg'] is None:
                all_slg_vals = [v for t_vals in [ts.get(t,{}).get('slg',[]) for t in ts] for v in t_vals]
                team_rolling[pfx]['slg'] = sum(all_slg_vals)/len(all_slg_vals) if all_slg_vals else 0.400

        # Batter individual stats and team stats calculation
        for pfx, batters in [('h', hb), ('a', ab)]:
            ops_list, avg_list, obp_list, slg_list = [], [], [], []
            team_ops_proxy = team_rolling[pfx]['ops']
            team_avg_proxy = team_rolling[pfx]['avg']
            team_obp_proxy = team_rolling[pfx]['obp']
            team_slg_proxy = team_rolling[pfx]['slg']

            for i in range(1, 10):
                if i <= len(batters):
                    b = batters[i-1]
                    b_avg = _safe_float(b.get('avg'), team_avg_proxy)
                    b_ops = _safe_float(b.get('ops'), team_ops_proxy)
                    b_obp = _safe_float(b.get('obp'), team_obp_proxy)
                    b_slg = _safe_float(b.get('slg'), team_slg_proxy)
                else:
                    b_avg = team_avg_proxy
                    b_ops = team_ops_proxy
                    b_obp = team_obp_proxy
                    b_slg = team_slg_proxy

                d[f'{pfx}_batter_{i}_avg'] = b_avg
                d[f'{pfx}_batter_{i}_ops'] = b_ops
                ops_list.append(b_ops)
                avg_list.append(b_avg)
                obp_list.append(b_obp)
                slg_list.append(b_slg)

            # Team stats = average of the starters (datos reales)
            d[f'{pfx}_ops'] = np.mean(ops_list)
            d[f'{pfx}_avg'] = np.mean(avg_list)
            d[f'{pfx}_obp'] = np.mean(obp_list)
            d[f'{pfx}_slg'] = np.mean(slg_list)

        # Calculate batter Savant features facing opposing pitcher's hand
        for pfx, batters, opp_hand in [('h', hb, ap_hand), ('a', ab, hp_hand)]:
            xwoba_vals, barrel_vals, hardhit_vals, ev_vals = [], [], [], []
            bat_speed_vals, swing_length_vals, sweetspot_vals, discipline_vals, efficiency_vals = [], [], [], [], []
            for i in range(1, 10):
                b_sav = None
                if i <= len(batters):
                    b = batters[i-1]
                    pid = b.get('personId', 0)
                    if pid > 0:
                        b_sav = get_batter_savant_stats(pid, date_str, opp_hand)
                
                xwoba_vals.append(b_sav['xwoba'] if b_sav and b_sav['xwoba'] is not None else 0.320)
                barrel_vals.append(b_sav['barrel_rate'] if b_sav and b_sav['barrel_rate'] is not None else 0.080)
                hardhit_vals.append(b_sav['hardhit_rate'] if b_sav and b_sav['hardhit_rate'] is not None else 0.400)
                ev_vals.append(b_sav['ev'] if b_sav and b_sav['ev'] is not None else 89.0)
                bat_speed_vals.append(b_sav['bat_speed'] if b_sav and b_sav['bat_speed'] is not None else 71.5)
                swing_length_vals.append(b_sav['swing_length'] if b_sav and b_sav['swing_length'] is not None else 7.3)
                sweetspot_vals.append(b_sav['sweetspot'] if b_sav and b_sav['sweetspot'] is not None else 0.33)
                discipline_vals.append(b_sav['discipline'] if b_sav and b_sav['discipline'] is not None else 1.5)
                efficiency_vals.append(b_sav['efficiency'] if b_sav and b_sav['efficiency'] is not None else 9.8)
                
            d[f'{pfx}_lineup_xwoba'] = np.mean(xwoba_vals)
            d[f'{pfx}_lineup_barrel'] = np.mean(barrel_vals)
            d[f'{pfx}_lineup_hardhit'] = np.mean(hardhit_vals)
            d[f'{pfx}_lineup_ev'] = np.mean(ev_vals)
            d[f'{pfx}_lineup_bat_speed'] = np.mean(bat_speed_vals)
            d[f'{pfx}_lineup_swing_length'] = np.mean(swing_length_vals)
            d[f'{pfx}_lineup_sweetspot'] = np.mean(sweetspot_vals)
            d[f'{pfx}_lineup_discipline'] = np.mean(discipline_vals)
            d[f'{pfx}_lineup_efficiency'] = np.mean(efficiency_vals)
            
            d[f'{pfx}_lineup_decision_quality'] = d[f'{pfx}_lineup_discipline'] * d[f'{pfx}_lineup_sweetspot'] * 10
            
        # Lineup Savant Differentials
        d['xwoba_lineup_diff'] = d['h_lineup_xwoba'] - d['a_lineup_xwoba']
        d['barrel_lineup_diff'] = d['h_lineup_barrel'] - d['a_lineup_barrel']
        d['hardhit_lineup_diff'] = d['h_lineup_hardhit'] - d['a_lineup_hardhit']
        d['bat_speed_diff'] = d['h_lineup_bat_speed'] - d['a_lineup_bat_speed']
        d['swing_length_diff'] = d['h_lineup_swing_length'] - d['a_lineup_swing_length']
        d['discipline_diff'] = d['h_lineup_discipline'] - d['a_lineup_discipline']
        d['efficiency_diff'] = d['h_lineup_efficiency'] - d['a_lineup_efficiency']
        
        # Matchup Advantages
        d['h_matchup_xwoba_diff'] = d['h_lineup_xwoba'] - d['a_starter_xwoba']
        d['a_matchup_xwoba_diff'] = d['a_lineup_xwoba'] - d['h_starter_xwoba']
        d['matchup_advantage'] = d['h_matchup_xwoba_diff'] - d['a_matchup_xwoba_diff']

        # Stuff+ Proxy
        for pfx in ['h', 'a']:
            spin = d[f'{pfx}_starter_spin']
            ext = d[f'{pfx}_starter_extension']
            ev_allowed = d[f'{pfx}_starter_ev']
            
            spin_z = (spin - 2250) / 250.0
            ext_z = (ext - 6.2) / 0.5
            ev_sup_z = (88.5 - ev_allowed) / 3.0
            
            d[f'{pfx}_starter_stuff_plus'] = (spin_z * 0.35) + (ext_z * 0.25) + (ev_sup_z * 0.40)

        # Context — extract real values from schedule API (single call, fix double-fetch bug)
        gd_ts=pd.to_datetime(date_str)
        season_start=pd.Timestamp(f'{gd_ts.year}-03-01')
        game_info = {}
        try:
            game_info = statsapi.get('game', {'gamePk': gid}).get('gameData', {})
        except Exception as e:
            logger.warning(f"Game info API failed for game {gid}: {e}")
        game_meta = game_info.get('game', {})
        d['series_game_number']=int(game_meta.get('series_game_number', game_meta.get('game_num', 1)))
        d['divisional_game']=int(TEAM_DIV.get(hn)==TEAM_DIV.get(an))
        d['park_factor']=PARK_FACTORS.get(hn, 1.00)
        # Rest days — Fix #29/#43: filter to current season, sort before taking last
        h_mask=(dfh['date']>=season_start)&((dfh['home_team']==hn)|(dfh['away_team']==hn))
        a_mask=(dfh['date']>=season_start)&((dfh['home_team']==an)|(dfh['away_team']==an))
        h_last=dfh.loc[h_mask,'date'].max() if h_mask.any() else None
        a_last=dfh.loc[a_mask,'date'].max() if a_mask.any() else None
        d['h_rest_days']=min((gd_ts-h_last).days,7) if h_last is not None and not pd.isna(h_last) else 1
        d['a_rest_days']=min((gd_ts-a_last).days,7) if a_last is not None and not pd.isna(a_last) else 1
        # Temperature and night game from weather/schedule API (fix 🟡7, reuse game_info)
        d['temperature']=70
        d['is_night']=1
        try:
            weather=game_info.get('weather',{})
            if weather.get('temp'): d['temperature']=int(weather['temp'])
            dt_str = game_meta.get('gameDate') or game_meta.get('game_datetime')
            if dt_str:
                dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%SZ')
                d['is_night'] = 1 if (dt.hour - 4) % 24 >= 17 else 0
        except Exception as e:
            logger.warning(f"Weather API failed for game {gid}, using default temp 70: {e}")
        # Diffs (from Prod)
        d['era_diff']=hs['era']-ast['era']; d['whip_diff']=hs['whip']-ast['whip']
        d['ops_diff']=d['h_ops']-d['a_ops']; d['avg_diff']=d['h_avg']-d['a_avg']
        d['slg_diff']=d['h_slg']-d['a_slg']; d['obp_diff']=d['h_obp']-d['a_obp']
        d['k_rate_diff']=d['h_starter_strikeout_rate']-d['a_starter_strikeout_rate']
        d['bb_rate_diff']=d['h_starter_walk_rate']-d['a_starter_walk_rate']
        hp_ops=[d.get(f'h_batter_{i}_ops',0.7) for i in range(1,10)]
        ap_ops=[d.get(f'a_batter_{i}_ops',0.7) for i in range(1,10)]
        d['lineup_power_diff']=np.mean(hp_ops)-np.mean(ap_ops)
        d['lineup_depth_diff']=np.mean([d.get(f'h_batter_{i}_avg',0.25) for i in range(5,10)])-\
                               np.mean([d.get(f'a_batter_{i}_avg',0.25) for i in range(5,10)])
        d['domination_diff']=(d['h_ops']-ast['whip'])-(d['a_ops']-hs['whip'])
        d['contact_q_diff']=(d['h_avg']*d['h_slg'])-(d['a_avg']*d['a_slg'])
        # SABER — NOTE: h_fip/a_fip is NOT real FIP (13*HR+3*BB-2*K)/IP+3.2.
        # This is a custom composite: ERA*0.6+WHIP*1.5+3.1. Models trained with this;
        # to use real FIP, retrain with corrected formula and update features.json.
        d['h_fip']=hs['era']*0.6+hs['whip']*1.5+3.1; d['a_fip']=ast['era']*0.6+ast['whip']*1.5+3.1
        d['h_tt_era']=hs['era']*0.4+d['h_fip']*0.6; d['a_tt_era']=ast['era']*0.4+d['a_fip']*0.6
        d['h_era_whip_ratio']=hs['era']/max(hs['whip'],0.1); d['a_era_whip_ratio']=ast['era']/max(ast['whip'],0.1)
        d['h_kbb']=hs['kbb']; d['a_kbb']=ast['kbb']
        d['h_dom']=(hs['strikeout_rate']-hs['walk_rate'])/max(hs['whip'],0.1)
        d['a_dom']=(ast['strikeout_rate']-ast['walk_rate'])/max(ast['whip'],0.1)
        d['h_era_x_a_ops']=hs['era']*d['a_ops']; d['a_era_x_h_ops']=ast['era']*d['h_ops']
        # Dynamic
        d['elo_diff']=elo_map.get(hn,1500)-elo_map.get(an,1500)
        hw=ts.get(hn,{}).get('wins',[]); aw=ts.get(an,{}).get('wins',[])
        d['h_streak']=sum(hw[-5:]); d['a_streak']=sum(aw[-5:])
        d['streak_diff']=d['h_streak']-d['a_streak']
        # Bullpen — differentiated metrics (fix 🟡6)
        rels=pph[pph['role']=='reliever']
        gd=pd.to_datetime(date_str)
        for tn,pfx in[(an,'a_'),(hn,'h_')]:
            tid=TEAM_IDS.get(tn)
            d[f'{pfx}bp_avail']=None; d[f'{pfx}bp_fatigue']=None; d[f'{pfx}bullpen_era_l3']=None
            if tid and not rels.empty:
                rec3 =rels[(rels['date']< gd)&(rels['date']>=gd-timedelta(days=3)) &(rels['team_id']==tid)]
                bp14 =rels[(rels['date']< gd)&(rels['date']>=gd-timedelta(days=14))&(rels['team_id']==tid)]
                bp30 =rels[(rels['date']< gd)&(rels['date']>=gd-timedelta(days=30))&(rels['team_id']==tid)]
                tot,tir=bp14['player_id'].nunique(),rec3['player_id'].nunique()
                if tot > 0:
                    d[f'{pfx}bp_avail']=max(0,tot-tir)/tot
                # bp_fatigue: prefer 14-day real ERA; fallback to 30-day; never a fixed constant
                # Sum er and ip instead of averaging era_game
                if not bp14.empty:
                    sum_er = sum(x.er if not pd.isna(x.er) else (x.runs or 0) for x in bp14.itertuples())
                    sum_ip = sum(parse_baseball_ip(x.ip) for x in bp14.itertuples())
                    d[f'{pfx}bp_fatigue'] = (sum_er * 9.0) / max(sum_ip, 0.1)
                elif not bp30.empty:
                    sum_er = sum(x.er if not pd.isna(x.er) else (x.runs or 0) for x in bp30.itertuples())
                    sum_ip = sum(parse_baseball_ip(x.ip) for x in bp30.itertuples())
                    d[f'{pfx}bp_fatigue'] = (sum_er * 9.0) / max(sum_ip, 0.1)
                
                if not rec3.empty:
                    sum_er = sum(x.er if not pd.isna(x.er) else (x.runs or 0) for x in rec3.itertuples())
                    sum_ip = sum(parse_baseball_ip(x.ip) for x in rec3.itertuples())
                    d[f'{pfx}bullpen_era_l3'] = (sum_er * 9.0) / max(sum_ip, 0.1)
                elif not bp14.empty:
                    sum_er = sum(x.er if not pd.isna(x.er) else (x.runs or 0) for x in bp14.itertuples())
                    sum_ip = sum(parse_baseball_ip(x.ip) for x in bp14.itertuples())
                    d[f'{pfx}bullpen_era_l3'] = (sum_er * 9.0) / max(sum_ip, 0.1)
        # If still None, use league-wide bullpen median from all teams in DB (real, not a constant)
        all_bp_era = pph[(pph['role']=='reliever') & pph['era_game'].notna()]['era_game']
        league_bp_era = float(all_bp_era.median()) if not all_bp_era.empty else 4.20
        league_bp_avail = 0.60  # fallback: only used if no team data at all
        for pfx in ['h_','a_']:
            if d[f'{pfx}bp_avail']  is None: d[f'{pfx}bp_avail']  = league_bp_avail
            if d[f'{pfx}bp_fatigue']    is None: d[f'{pfx}bp_fatigue']    = league_bp_era
            if d[f'{pfx}bullpen_era_l3'] is None: d[f'{pfx}bullpen_era_l3'] = league_bp_era
        # ── NEW POWER FEATURES ──
        h_rf = sum(ts.get(hn,{}).get('runs_for',[])[-30:]); h_ra = sum(ts.get(hn,{}).get('runs_ag',[])[-30:])
        a_rf = sum(ts.get(an,{}).get('runs_for',[])[-30:]); a_ra = sum(ts.get(an,{}).get('runs_ag',[])[-30:])
        d['h_pyth_pct'] = h_rf**2 / (h_rf**2 + h_ra**2 + 1e-9)
        d['a_pyth_pct'] = a_rf**2 / (a_rf**2 + a_ra**2 + 1e-9)
        d['pyth_diff'] = d['h_pyth_pct'] - d['a_pyth_pct']
        h_rd = [f-a2 for f,a2 in zip(ts.get(hn,{}).get('runs_for',[])[-10:], ts.get(hn,{}).get('runs_ag',[])[-10:])]
        a_rd = [f-a2 for f,a2 in zip(ts.get(an,{}).get('runs_for',[])[-10:], ts.get(an,{}).get('runs_ag',[])[-10:])]
        d['h_run_diff_10'] = sum(h_rd)/max(len(h_rd),1)
        d['a_run_diff_10'] = sum(a_rd)/max(len(a_rd),1)
        d['run_diff_diff'] = d['h_run_diff_10'] - d['a_run_diff_10']
        hw_rec = ts.get(hn,{}).get('home_wins',[])
        aw_rec = ts.get(an,{}).get('away_wins',[])
        d['h_home_wpct'] = sum(hw_rec[-20:])/max(len(hw_rec[-20:]),1)
        d['a_away_wpct'] = sum(aw_rec[-20:])/max(len(aw_rec[-20:]),1)
        h_w_a = h2h_wins.get((hn,an),0); a_w_h = h2h_wins.get((an,hn),0)
        d['h2h_advantage'] = (h_w_a - a_w_h)/max(h_w_a + a_w_h, 1)
        h_hist = starter_era_hist.get(hp_id, []); a_hist = starter_era_hist.get(ap_id, [])
        d['h_starter_era_l3'] = sum(h_hist[-3:])/len(h_hist[-3:]) if h_hist else hs['era']
        d['a_starter_era_l3'] = sum(a_hist[-3:])/len(a_hist[-3:]) if a_hist else ast['era']
        # ── VOLATILITY FEATURES (starter ERA std, sample IP, injury flag) ──
        starters_pp = pph[pph['role']=='starter'] if not pph.empty else pd.DataFrame()
        for pid, pfx in [(hp_id, 'h'), (ap_id, 'a')]:
            try:
                if not starters_pp.empty:
                    season_start = pd.Timestamp(f'{gd.year}-03-01')
                    p_starts = starters_pp[(starters_pp['player_id']==pid)&(starters_pp['date']>=season_start)&(starters_pp['date']<gd)]
                    p_starts = p_starts.sort_values('date')
                    if len(p_starts) >= 3:
                        eras = p_starts['era_game'].dropna().tolist()
                        recent7 = eras[-7:]
                        d[f'{pfx}_starter_era_std'] = float(np.std(recent7)) if len(recent7) >= 3 else 3.50
                        total_ip = sum(parse_baseball_ip(x) for x in p_starts['ip'])
                        d[f'{pfx}_starter_sample_ip'] = total_ip
                        dates = p_starts['date'].tolist()
                        if len(dates) >= 2:
                            gap = (dates[-1] - dates[-2]).days
                            d[f'{pfx}_starter_injury_flag'] = 1 if gap > 15 else 0
                        else:
                            d[f'{pfx}_starter_injury_flag'] = 0
                    else:
                        d[f'{pfx}_starter_era_std'] = 3.50
                        d[f'{pfx}_starter_sample_ip'] = sum(parse_baseball_ip(x) for x in p_starts['ip']) if not p_starts.empty else 0
                        d[f'{pfx}_starter_injury_flag'] = 0
                else:
                    d[f'{pfx}_starter_era_std'] = 3.50
                    d[f'{pfx}_starter_sample_ip'] = 0
                    d[f'{pfx}_starter_injury_flag'] = 0
            except:
                d[f'{pfx}_starter_era_std'] = 3.50
                d[f'{pfx}_starter_sample_ip'] = 0
                d[f'{pfx}_starter_injury_flag'] = 0
        h_top4 = np.mean(hp_ops[:min(4,len(hp_ops))]); h_bot5 = np.mean(hp_ops[min(4,len(hp_ops)):])
        a_top4 = np.mean(ap_ops[:min(4,len(ap_ops))]); a_bot5 = np.mean(ap_ops[min(4,len(ap_ops)):])
        d['lineup_quality_weighted_diff'] = (h_top4*0.6 + h_bot5*0.4) - (a_top4*0.6 + a_bot5*0.4)
        d['momentum_h'] = d['h_streak'] * (np.clip(d['elo_diff'], -200, 200)/200.0)
        d['momentum_a'] = d['a_streak'] * (-np.clip(d['elo_diff'], -200, 200)/200.0)
        h_lev = d['h_bp_avail'] / max(d['h_bp_fatigue'], 0.5)
        a_lev = d['a_bp_avail'] / max(d['a_bp_fatigue'], 0.5)
        d['bullpen_leverage_diff'] = h_lev - a_lev
        d['game_pk'] = gid
        d['date'] = pd.to_datetime(date_str)
        d['pp_h_starter_id'] = hp_id
        d['pp_a_starter_id'] = ap_id
        today_dicts.append(d)
        game_data.append((g, hn, an, hp_id, ap_id, hp, ap, confirmed, lineup_confirmed, hb, ab, scan, pitcher_data_complete, hs, ast))

    if today_dicts:
        # Build today's rows
        df_today = pd.DataFrame(today_dicts)
        df_today['h_runs_total'] = np.nan
        df_today['a_runs_total'] = np.nan

        # Columns that build_all_features needs to execute successfully:
        keep_cols = {
            'game_pk', 'date', 'home_team', 'away_team', 
            'h_starter', 'a_starter', 'pp_h_starter_id', 'pp_a_starter_id',
            'h_starter_era', 'h_starter_whip', 'h_starter_strikeout_rate', 'h_starter_walk_rate',
            'a_starter_era', 'a_starter_whip', 'a_starter_strikeout_rate', 'a_starter_walk_rate',
            'h_runs_total', 'a_runs_total'
        }
        
        # Save all other columns of df_today to restore them later
        restore_data = {}
        for col in df_today.columns:
            if col not in keep_cols:
                restore_data[col] = df_today[col].copy()
                
        # Drop those columns from df_today so they don't cause duplicate conflicts in build_all_features
        df_today_clean = df_today[list(keep_cols & set(df_today.columns))].copy()

        df_db = dfh.reset_index(drop=True)
        pp_db = pph.reset_index(drop=True)

        combined_df = pd.concat([df_db, df_today_clean], ignore_index=True)

        print(f"🧬 Computando features v4 (Markov, MC, ELO, Stuff+) para {len(today_dicts)} juegos...")
        df_all, feats, target = build_all_features(combined_df, pp_db, predict_mode=True, elo_sys=elo_sys)

        # Extract today's processed rows using game_pk
        today_game_pks = [gd['game_id'] for gd, *rest in game_data]
        df_today_processed = df_all[df_all['game_pk'].isin(today_game_pks)].copy()

        # Restore ONLY columns that build_all_features did NOT compute
        # (everything in feats was computed by build_all_features and must not be overridden)
        feats_set = set(feats)
        for col, orig_series in restore_data.items():
            if col in feats_set:
                continue
            mapping = df_today.set_index('game_pk')[col]
            df_today_processed[col] = df_today_processed['game_pk'].map(mapping)

        processed_rows = {}
        for _, row in df_today_processed.iterrows():
            gpk = int(row['game_pk'])
            row_dict = row.to_dict()
            for f in feats:
                if f not in row_dict or pd.isna(row_dict[f]):
                    row_dict[f] = 0
            processed_rows[gpk] = row_dict

        # Second loop: perform predictions and compile results
        for g, hn, an, hp_id, ap_id, hp, ap, confirmed, lineup_confirmed, hb, ab, scan, pitcher_data_complete, hs, ast in game_data:
            gid = g['game_id']
            d = processed_rows.get(gid)
            if not d:
                continue

            if USE_V3_ENSEMBLE:
                r = predict_v3(d)
                # the v3 function already maps output keys properly
            else:
                r = ensemble_predict(models, scalers, feats, d, cal)

            pick = hn if r['prob'] >= 0.5 else an
            # 🚫 NEVER bet on or against the Angels
            angels_rule = "Los Angeles Angels" in (hn, an)
            conf_raw = r['conf_raw']
            conf_cal = r['conf_calibrated']
            if not pitcher_data_complete:
                conf_cal = min(conf_cal, 64.9)
                conf_raw = min(conf_raw, 64.9)
            cons = r['consensus']
            god = r['god_mode']

            # BETA — cannot cross tier boundaries (fix 🔵10)
            beta_tag = ""
            if beta_analyzer and confirmed:
                try:
                    ps = beta_analyzer.analyze_prescan(hp_id, ap_id)
                    cd = None
                    sup = ps.get('elo_starter_diff', 0) / 100
                    if lineup_confirmed and len(hb) >= 7:
                        cd = beta_analyzer.analyze_confirmed(hb, ab, hp_id, ap_id)
                        sup = beta_analyzer.compute_superiority(ps, cd)
                        delta = beta_analyzer.compute_confidence_delta(ps, cd)
                    else:
                        delta = beta_analyzer.compute_confidence_delta(ps)
                    if abs(sup) > 0.12:
                        arr = "▲" if sup > 0 else "▼"
                        beta_tag = f"🧬{arr}{abs(sup):.0%}"
                        if pick == an: delta = -delta
                        conf_cal = min(max(conf_cal + delta * 100, 50), 99)
                except Exception as e:
                    pass

            # Stability alerts check
            h_std = d.get('h_starter_era_std', 3.50)
            a_std = d.get('a_starter_era_std', 3.50)
            h_ip = d.get('h_starter_sample_ip', 0.0)
            a_ip = d.get('a_starter_sample_ip', 0.0)
            h_inj = d.get('h_starter_injury_flag', 0)
            a_inj = d.get('a_starter_injury_flag', 0)

            pick_is_home = (pick == hn)
            p_streak = d['h_streak'] if pick_is_home else d['a_streak']
            opp_streak = d['a_streak'] if pick_is_home else d['h_streak']
            p_std = h_std if pick_is_home else a_std

            is_hot = p_streak >= 4 and opp_streak <= 2
            is_cold = p_streak <= 2 and opp_streak >= 4
            is_high_vol = p_std > 6.0

            form_vol_flags = []
            if angels_rule:
                form_vol_flags.append("🚫 ANGELS")
            if use_filters:
                if is_high_vol:
                    conf_cal -= 3.5
                    conf_cal = min(conf_cal, 74.9)
                    form_vol_flags.append("⚠️ ALTA VOL")
                if is_cold and conf_cal >= 78 and conf_cal < 80:
                    conf_cal -= 4.6
                if is_hot:
                    form_vol_flags.append("🔥 CALIENTE")
                elif is_cold:
                    form_vol_flags.append("❄️ FRÍO")

            # TIER based on calibrated confidence
            if god and conf_cal >= 80:
                tier = "💎 GOD"
                if is_cold:
                    tier = "💎 GOD (❄️ TRAMPA)"
            elif conf_cal >= 78:
                tier = "🎯 SNIPER"
            elif conf_cal >= 65:
                tier = "📦 VOLUMEN"
            else:
                tier = "📊 STANDARD"

            # ── Prediction Quality Score (0-10) ──
            pq_score = 0
            pq_score += min(max(cons - 1, 0), 3)
            pick_era = d['h_starter_era'] if pick_is_home else d['a_starter_era']
            pick_bp = d['h_bp_avail'] if pick_is_home else d['a_bp_avail']
            if pick_era < 3.2 and pick_bp > 0.65:
                pq_score += 2
            elif pick_era < 3.8:
                pq_score += 1
            xwoba_diff = d.get('xwoba_lineup_diff', 0)
            if pick_is_home and xwoba_diff > 0.015:
                pq_score += 1
            elif not pick_is_home and xwoba_diff < -0.015:
                pq_score += 1
            if p_std < 2.5:
                pq_score += 1
            elif p_std > 5.0:
                pq_score -= 2
            if god:
                pq_score += 2
            pq_score = max(0, min(10, pq_score))

            tc = {" GOD": C['P'], "NIPER": C['R'], "LUMEN": C['Y']}.get(tier[3:7], C['D'])
            wp = r.get('window_probs', {})

            base_units = {
                "💎 GOD": 2.0,
                "💎 GOD (❄️ TRAMPA)": 2.0,
                "🎯 SNIPER": 1.5,
                "📦 VOLUMEN": 1.0,
                "📊 STANDARD": 0.5
            }.get(tier, 0.5)

            recommended_units = base_units
            if angels_rule:
                recommended_units = 0.0
            if use_filters:
                if is_hot:
                    recommended_units *= 2.0
                if is_cold and is_high_vol:
                    recommended_units = 0.0
                    form_vol_flags.append("❌ EVITAR")
                elif is_cold or is_high_vol:
                    recommended_units *= 0.5

            warnings_list = []
            h_reasons = []
            if h_std > 3.2: h_reasons.append(f"Alta Volatilidad (std={h_std:.2f})")
            if h_ip < 20.0: h_reasons.append(f"Muestra Pequeña (ip={h_ip:.1f})")
            if h_inj == 1: h_reasons.append("Regreso Reciente IL")
            if h_reasons:
                warnings_list.append(f"{hp} ({hn}): " + ", ".join(h_reasons))

            a_reasons = []
            if a_std > 3.2: a_reasons.append(f"Alta Volatilidad (std={a_std:.2f})")
            if a_ip < 20.0: a_reasons.append(f"Muestra Pequeña (ip={a_ip:.1f})")
            if a_inj == 1: a_reasons.append("Regreso Reciente IL")
            if a_reasons:
                warnings_list.append(f"{ap} ({an}): " + ", ".join(a_reasons))

            if form_vol_flags:
                pick = f"{pick} ({' '.join(form_vol_flags)})"

            game_key = f"{date_str}_{hn}_{an}"
            momio_pct, momio_emoji = estimate_momio_change(
                game_key, conf_cal, scan, lineup_confirmed, hp, ap, hs, ast,
                team_name=hn, team_id=TEAM_IDS.get(hn), date_str=date_str,
                lineup_batters=hb)

            results.append({
                'home': hn, 'away': an, 'pick': pick, 'conf_raw': conf_raw,
                'conf_cal': conf_cal, 'tier': tier, 'god': god, 'cons': cons, 'windows': wp,
                'h_starter': hp, 'a_starter': ap,
                'h_vol': {'std': h_std, 'ip': h_ip, 'injury': h_inj},
                'a_vol': {'std': a_std, 'ip': a_ip, 'injury': a_inj},
                'stability_warnings': warnings_list,
                'sc': sc, 'scan': scan, 'tc': tc, 'total_windows': r['total_windows'], 'beta_tag': beta_tag,
                'recommended_units': recommended_units,
                'momio_pct': momio_pct, 'momio_emoji': momio_emoji,
                'pq_score': pq_score
            })
    results.sort(key=lambda x: x['conf_cal'], reverse=True)
    tiers_order = ["💎 GOD", "💎 GOD (❄️ TRAMPA)", "🎯 SNIPER", "📦 VOLUMEN", "📊 STANDARD"]
    
    from rich.console import Console
    from rich.table import Table
    import shutil
    
    width, _ = shutil.get_terminal_size(fallback=(120, 24))
    console = Console(width=max(width, 130), force_terminal=True)
    
    tier_styles = {
        "💎 GOD": {"header": "bold magenta", "border": "magenta", "pick": "bold magenta"},
        "💎 GOD (❄️ TRAMPA)": {"header": "bold magenta", "border": "magenta", "pick": "bold white on red"},
        "🎯 SNIPER": {"header": "bold red", "border": "red", "pick": "bold red"},
        "📦 VOLUMEN": {"header": "bold yellow", "border": "yellow", "pick": "bold yellow"},
        "📊 STANDARD": {"header": "bold blue", "border": "blue", "pick": "bold white"}
    }
    
    hidden = []; shown = results
    if hidden:
        print(f"\n[dim]{'─'*60}[/]")
        print(f"[dim]⏭️  {len(hidden)} picks ocultos por Quality Score < 6[/]")
    for t_target in tiers_order:
        tier_results = [r for r in shown if r['tier'] == t_target]
        if not tier_results: continue
        
        style_info = tier_styles.get(t_target, {"header": "bold white", "border": "white", "pick": "bold white"})
        
        table = Table(
            title=f"\n=== {t_target} ===",
            title_style=style_info["header"],
            show_lines=True,
            border_style=style_info["border"],
            expand=False,
            width=min(console.width, 180)
        )

        table.add_column("MATCHUP", style="bold white", no_wrap=True, width=26)
        table.add_column("PICK", style=style_info["pick"], no_wrap=True, width=18)
        table.add_column("CONFIANZA", justify="center", no_wrap=True, width=11)
        table.add_column("APUESTA", justify="center", no_wrap=True, width=11)
        table.add_column("MOMIO", justify="center", no_wrap=True, width=11)
        table.add_column("JUSTIFICACIÓN", style="cyan", width=42)
        table.add_column("⚠️ RIESGOS", style="yellow", width=38)

        for r_item in tier_results:
            scan_val = r_item['scan']
            if "CONFIRMED" in scan_val:
                scan_icon = "✅"
            elif "GUESS" in scan_val:
                scan_icon = "🔮"
            else:
                scan_icon = "📡"

            matchup_str = f"{scan_icon} {r_item['away'][:12]} @\n   {r_item['home'][:12]}"

            conf_raw = r_item['conf_raw']
            conf_cal = r_item['conf_cal']
            edge = conf_cal - 50
            if edge >= 35:
                conf_color = "bright_green"
            elif edge >= 20:
                conf_color = "green"
            elif edge >= 10:
                conf_color = "yellow"
            else:
                conf_color = "white"
            conf_str = f"[{conf_color}]{conf_cal:.0f}%[/]\n[dim](raw {conf_raw:.0f}%)[/]"

            units_val = r_item.get('recommended_units', 0.5)
            if units_val == 0.0:
                units_str = "[bold red]0.00 U[/]\n[dim]EVITAR[/]"
            elif units_val >= 3.0:
                units_str = f"[bold bright_green]{units_val:.2f} U[/]\n[green]FUERTE[/]"
            elif units_val >= 1.5:
                units_str = f"[bold green]{units_val:.2f} U[/]\n[green]SÓLIDA[/]"
            elif units_val >= 0.75:
                units_str = f"[bold yellow]{units_val:.2f} U[/]\n[yellow]OK[/]"
            else:
                units_str = f"[white]{units_val:.2f} U[/]\n[dim]PEQUEÑA[/]"

            momio_emoji = r_item['momio_emoji']
            momio_pct = r_item['momio_pct']
            if momio_pct > 0:
                momio_str = f"[green]{momio_emoji} +{momio_pct:.1f}%[/]\n[dim]línea bajó[/]"
            elif momio_pct < 0:
                momio_str = f"[red]{momio_emoji} {momio_pct:.1f}%[/]\n[dim]línea subió[/]"
            else:
                momio_str = f"[white]— 0.0%[/]\n[dim]sin cambio[/]"

            just_parts = []
            cons_str = f"[bold]{r_item['cons']}/{r_item['total_windows']}[/]"
            just_parts.append(f"Consenso: {cons_str}")
            if r_item.get('beta_tag'):
                just_parts.append(f"Tag: {r_item['beta_tag']}")
            hp = r_item.get('h_starter', '') or 'TBD'
            ap = r_item.get('a_starter', '') or 'TBD'
            just_parts.append(f"ST: {ap} vs {hp}")
            just_str = "\n".join(just_parts)

            if r_item['stability_warnings']:
                risk_str = "\n".join(f"⚠ {w}" for w in r_item['stability_warnings'])
            else:
                risk_str = "[green]✓ Sin riesgos[/]"

            table.add_row(
                matchup_str,
                r_item['pick'][:18],
                conf_str,
                units_str,
                momio_str,
                just_str,
                risk_str
            )
            
        console.print(table)

    gm = sum(1 for r in shown if 'GOD' in r['tier'])
    sn = sum(1 for r in shown if 'SNIPER' in r['tier'])
    vol = sum(1 for r in shown if 'VOLUMEN' in r['tier'])
    std = sum(1 for r in shown if 'STANDARD' in r['tier'])
    console.print(f"\n[bold]Total:[/] [bold magenta]{gm} GOD[/] | [bold red]{sn} SNIPER[/] | [bold yellow]{vol} VOLUMEN[/] | [bold blue]{std} STANDARD[/]")
    trap_teams = {"Los Angeles Angels", "Washington Nationals", "Boston Red Sox", "Houston Astros", "St. Louis Cardinals", "Athletics", "Oakland Athletics"}
    total_played = sum(1 for r in shown if r['recommended_units'] > 0)
    ruined = sum(1 for r in shown if r['recommended_units'] > 0 and (r['home'] in trap_teams or r['away'] in trap_teams))
    if ruined > 0:
        console.print(f"[bold]Picks jugados:[/] {total_played} | [bold red]🚫 Arruinados (trampa): {ruined}[/]")
    return results

def backtest_clean():
    """
    Walk-forward backtest with NO data leakage.
    For each year split: filter data to only games up to test_year,
    build features on that subset ONLY, then train/test split.
    """
    print("="*90+"\n OMEGA v3 BACKTEST CLEAN (walk-forward, no leak)\n"+"="*90)
    df,pp=load_data()
    df = df.reset_index(drop=True)
    
    splits=[(2023,2024),(2024,2025),(2025,2026)]
    results=[]
    for ty,ey in splits:
        print(f"\n--- Split: train ≤{ty}, test {ey} ---")
        # Use ONLY data up to test year — zero future visibility
        mask_up_to_test = df['date'].dt.year <= ey
        df_split = df[mask_up_to_test].copy()
        if len(df_split) < 200:
            print(f"  Not enough data ({len(df_split)} games), skipping")
            continue
        
        # Build features on this subset ONLY
        df_feat, feats, target = build_all_features(df_split, pp)
        
        # Now split: train up to ty, test on ey
        trm = df_feat['date'].dt.year <= ty
        tem = df_feat['date'].dt.year == ey
        dtr = df_feat[trm]
        dte = df_feat[tem]
        tgt_tr = target[trm.values]
        tgt_te = target[tem.values]
        
        if len(dtr) < 100 or len(dte) < 10:
            print(f"  Train={len(dtr)} Test={len(dte)} — too small, skip")
            continue
        
        print(f"  Train: {len(dtr)} games, Test: {len(dte)} games")
        models,scalers,cal=train_multiwindow(dtr,feats,tgt_tr,save=False)
        
        from models.game import WEIGHTS
        X_te = dte[feats].fillna(0).values
        if 'all' not in scalers:
            print(f"  No 'all' scaler, skipping")
            continue
        X_te_s = scalers['all'].transform(X_te)
        
        probs = np.zeros(len(X_te))
        indiv = []
        for mname, weight in WEIGHTS.items():
            key = f'all_{mname}'
            if key not in models: continue
            p = models[key].predict_proba(X_te_s)[:,1]
            probs += p * weight
            indiv.append(p)
        probs /= sum(WEIGHTS.values())
        
        god_mask = np.zeros(len(X_te), dtype=bool)
        if len(indiv) >= 5:
            indiv_arr = np.array(indiv)
            god_mask = np.all(indiv_arr > 0.65, axis=0) | np.all(indiv_arr < 0.35, axis=0)
        
        cons_arr = np.ones(len(X_te), dtype=int)
        for wname in ['w50','w25','w10']:
            if wname not in scalers: continue
            X_w = scalers[wname].transform(X_te)
            wp = np.zeros(len(X_te))
            for mname, weight in WEIGHTS.items():
                key = f'{wname}_{mname}'
                if key not in models: continue
                wp += models[key].predict_proba(X_w)[:,1] * weight
            wp /= sum(WEIGHTS.values())
            cons_arr += ((wp >= 0.5) == (probs >= 0.5)).astype(int)
        
        picks = (probs >= 0.5).astype(int)
        conf_raw = np.maximum(probs, 1-probs) * 100
        for i in range(len(tgt_te)):
            cr = round(conf_raw[i], 1)
            bucket = f"{int(cr//5)*5}_{int(cr//5)*5+5}"
            cc = cal.get(bucket, cr) if cal else cr
            
            row = dte.iloc[i]
            pick_is_home = (picks[i] == 1)
            p_streak = row.get('h_streak', 0) if pick_is_home else row.get('a_streak', 0)
            opp_streak = row.get('a_streak', 0) if pick_is_home else row.get('h_streak', 0)
            p_std = row.get('h_starter_era_std', 3.5) if pick_is_home else row.get('a_starter_era_std', 3.5)
            
            is_hot = p_streak >= 4 and opp_streak <= 2
            is_cold = p_streak <= 2 and opp_streak >= 4
            is_high_vol = p_std > 6.0
            
            if is_high_vol:
                cc -= 3.5
                cc = min(cc, 74.9)
            if is_cold and cc >= 78 and cc < 80:
                cc -= 4.6
            
            results.append({'year':ey,'prob':probs[i],'conf_raw':cr,
                          'conf_cal':round(cc,1),'god':bool(god_mask[i]),
                          'consensus':int(cons_arr[i]),'pick':int(picks[i]),
                          'real':int(tgt_te[i]),'hit':int(picks[i]==tgt_te[i])})
    
    if not results:
        print("No results generated")
        return
    
    R=pd.DataFrame(results)
    acc=R['hit'].mean()
    print(f"\n{'='*60}")
    print(f"  CLEAN BACKTEST — WALK-FORWARD (NO LEAK)")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.2%} ({R['hit'].sum()}/{len(R)})")
    profit=((R['hit']*1.91)-1).sum()
    print(f"  Profit (1u @ 1.91): {profit:+.2f}u")
    
    conf_col = 'conf_cal'
    def get_tier(row):
        if row['god'] and row[conf_col] >= 80: return "GOD"
        elif row[conf_col] >= 78: return "SNIPER"
        elif row[conf_col] >= 65: return "VOLUMEN"
        else: return "STANDARD"
    R['tier'] = R.apply(get_tier, axis=1)
    
    print(f"\n  BY TIER:")
    for t in ["GOD", "SNIPER", "VOLUMEN", "STANDARD"]:
        subset = R[R['tier'] == t]
        if len(subset) > 0:
            print(f"    {t}: {subset['hit'].mean():.2%} ({subset['hit'].sum()}/{len(subset)})")
    
    for c in [4,3]:
        sc=R[R['consensus']>=c]
        if len(sc)>=5:
            print(f"    Consensus ≥{c}: {sc['hit'].mean():.2%} ({sc['hit'].sum()}/{len(sc)})")
    
    for y in sorted(R['year'].unique()):
        yr=R[R['year']==y]
        n_conf=len(yr)
        print(f"    {y}: {yr['hit'].mean():.2%} ({yr['hit'].sum()}/{n_conf})")
    
    R.to_csv('backtest_clean_results.csv', index=False)
    print(f"\n  Resultados guardados: backtest_clean_results.csv")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'backtest', 'backtest_clean'], default='predict')
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--beta', action='store_true')
    parser.add_argument('--skip_sync', action='store_true')
    args = parser.parse_args()
    
    if args.mode == 'predict':
        predict_live(date_str=args.date, beta=args.beta, skip_sync=args.skip_sync)
    elif args.mode == 'backtest':
        backtest(beta=args.beta)
    elif args.mode == 'backtest_clean':
        backtest_clean()
