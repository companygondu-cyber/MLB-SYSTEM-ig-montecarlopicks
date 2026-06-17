"""
🧬 OMEGA BETA v1.0 — Player ELO + Lineup Superiority
Modulo activable con --beta en omega_final.py

2 tiers:
  PRE-SCAN  → pitcher ELO diff, bullpen ELO, starter trend
  CONFIRMED → + lineup ELO superiority, depth, hot batters
"""
import os, sys, sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

TEAM_IDS = {
    "Arizona Diamondbacks":109,"Atlanta Braves":144,"Baltimore Orioles":110,
    "Boston Red Sox":111,"Chicago Cubs":112,"Chicago White Sox":145,
    "Cincinnati Reds":113,"Cleveland Guardians":114,"Colorado Rockies":115,
    "Detroit Tigers":116,"Houston Astros":117,"Kansas City Royals":118,
    "Los Angeles Angels":108,"Los Angeles Dodgers":119,"Miami Marlins":146,
    "Milwaukee Brewers":158,"Minnesota Twins":142,"New York Mets":121,
    "New York Yankees":147,"Oakland Athletics":133,"Philadelphia Phillies":143,
    "Pittsburgh Pirates":134,"San Diego Padres":135,"San Francisco Giants":137,
    "Seattle Mariners":136,"St. Louis Cardinals":138,"Tampa Bay Rays":139,
    "Texas Rangers":140,"Toronto Blue Jays":141,"Washington Nationals":120,
}

def _find_db():
    for p in [os.path.join(os.path.dirname(__file__), 'data', 'omega_2026_BETA.db'),
              os.path.join(os.path.dirname(__file__), 'data', 'omega_2026.db'),
              '/Users/rendon/Desktop/OmegaFinal/data/omega_2026_BETA.db']:
        if os.path.exists(p): return p
    return None

DB_PATH = _find_db()
if DB_PATH is None:
    import warnings
    warnings.warn('omega_beta: No database found. BETA module will be non-functional.')

BETA_K_PITCHER = 32
BETA_K_BATTER = 24
BETA_K_BP = 16

class PlayerELO:
    def __init__(self):
        self.pitcher_elo = {}   # player_id -> elo
        self.pitcher_elo_history = {} # (player_id, game_pk) -> elo prior to game
        self.batter_elo = {}    # player_id -> elo
        self.batter_elo_history = {} # (player_id, game_pk) -> elo prior to game
        self.recent_form = {}   # player_id -> [perf_scores]
        self.pitcher_names = {} # lowercase_name -> player_id
        self.batter_names = {}  # lowercase_name -> player_id
        self.load()

    def load(self):
        if DB_PATH is None:
            return
        conn = sqlite3.connect(DB_PATH)
        # Pitcher ELO desde pitcher_performances (orden cronologico)
        pp = pd.read_sql("SELECT * FROM pitcher_performances ORDER BY date, game_pk", conn)
        conn.close()
        pp['date'] = pd.to_datetime(pp['date'], format='mixed', errors='coerce')
        if pp.empty:
            return
        
        # Build name-to-id mapping (including partial name matching) from pitchers
        for row in pp.to_dict('records'):
            name = str(row.get('player_name', '')).strip().lower()
            pid = int(row['player_id'])
            if name and pid > 0:
                self.pitcher_names[name] = pid
                parts = name.replace(',', '').split()
                for p in parts:
                    if len(p) > 3:
                        self.pitcher_names[p] = pid
        # Build batter name mapping from batter_performances + players_master
        conn2 = sqlite3.connect(DB_PATH)
        tables = [r[0] for r in conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        has_bp = 'batter_performances' in tables
        has_pm = 'players_master' in tables
        bp = pd.read_sql("SELECT DISTINCT player_id, player_name FROM batter_performances", conn2) if has_bp else pd.DataFrame()
        pm = pd.read_sql("SELECT player_id, player_name FROM players_master", conn2) if has_pm else pd.DataFrame()
        conn2.close()
        for row in pd.concat([bp, pm]).drop_duplicates(subset=['player_id']).to_dict('records'):
            name = str(row.get('player_name', '')).strip().lower()
            pid = int(row['player_id']) if row['player_id'] else 0
            if name and pid > 0:
                self.batter_names[name] = pid
                parts = name.replace(',', '').split()
                for p in parts:
                    if len(p) > 3:
                        self.batter_names[p] = pid
        
        for row in pp.to_dict('records'):
            pid = int(row['player_id'])
            ip = max(parse_baseball_ip(row['ip']), 0.1)
            er = int(row['er'] or 0)
            k = int(row['k'] or 0)
            bb = int(row.get('bb', 0) or 0)
            h = int(row.get('hits', 0) or 0)
            era_game = round(er * 9 / ip, 2)
            whip = round((h + bb) / ip, 2)
            # Fix 🔴2: Game score normalized to [0, 1] range
            # Good outing (7IP, 1ER, 8K, 1BB) → ~0.75
            # Bad outing (3IP, 5ER, 2K, 4BB) → ~0.15
            # Average outing (5IP, 3ER, 5K, 2BB) → ~0.45
            raw_score = 50 + k * 1.0 - bb * 1.0 - er * 2.0 - h * 0.5 + ip * 2.0
            perf = max(0.0, min(1.0, raw_score / 100.0))  # Normalized [0, 1]
            # Expected performance: ELO-based expectation, also [0, 1]
            expected = self.pitcher_elo.get(pid, 1500)
            self.pitcher_elo_history[(pid, int(row['game_pk']))] = expected
            exp_perf = max(0.0, min(1.0, (expected - 1000) / 1000.0))  # 1000→0, 1500→0.5, 2000→1
            k_factor = BETA_K_PITCHER if row['role'] == 'starter' else BETA_K_BP
            delta = k_factor * (perf - exp_perf)  # Now both [0,1], delta max = ±32
            self.pitcher_elo[pid] = max(800, min(2200, expected + delta))
            if pid not in self.recent_form: self.recent_form[pid] = []
            self.recent_form[pid].append(perf)

        # Batter ELO desde batter_performances
        conn = sqlite3.connect(DB_PATH)
        try:
            bp = pd.read_sql("SELECT * FROM batter_performances ORDER BY date, game_pk", conn)
        except Exception:
            bp = pd.DataFrame()
        conn.close()
        bp['date'] = pd.to_datetime(bp['date'], format='mixed', errors='coerce')
        if bp.empty:
            return
        for row in bp.to_dict('records'):
            pid = int(row['player_id'])
            # Fix #38: Don't use 'or' — 0.0 OPS is valid
            _ops = row.get('ops', None)
            ops = float(_ops) if _ops is not None and _ops != '' else 0.700
            _avg = row.get('avg', None)
            avg = float(_avg) if _avg is not None and _avg != '' else 0.250
            hits = int(row.get('hits', 0) or 0)
            bb = int(row.get('bb', 0) or 0)
            pa = max(int(row.get('plate_appearances', 0) or 0), 1)
            # Fix #39: Normalize perf to [0, 1] like pitchers
            # OPS range ~[0, 1.5], typical ~0.7. Map to [0, 1].
            ops_norm = max(0.0, min(1.0, ops / 1.2))  # 0→0, 0.6→0.5, 1.2→1.0
            walk_bonus = min((bb / pa) * 2, 0.2)  # Max 0.2 bonus
            perf = max(0.0, min(1.0, ops_norm + walk_bonus))
            expected = self.batter_elo.get(pid, 1500)
            self.batter_elo_history[(pid, int(row['game_pk']))] = expected
            exp_perf = max(0.0, min(1.0, (expected - 1000) / 1000.0))  # Same scale as pitchers
            delta = BETA_K_BATTER * (perf - exp_perf)  # Max ±24
            self.batter_elo[pid] = max(800, min(2200, expected + delta))

    def get_id_by_name(self, name, mapping):
        if not name or not mapping: return 0
        key = name.strip().lower()
        if key in mapping:
            return mapping[key]
        for part in key.split():
            if len(part) > 3 and part in mapping:
                return mapping[part]
        return 0

    def get_pitcher_id_by_name(self, name):
        return self.get_id_by_name(name, self.pitcher_names)

    def get_batter_id_by_name(self, name):
        return self.get_id_by_name(name, self.batter_names)

    def get_pitcher_elo(self, player_id):
        return self.pitcher_elo.get(int(player_id), 1500)

    def get_historical_pitcher_elo(self, player_id, game_pk):
        return self.pitcher_elo_history.get((int(player_id), int(game_pk)), self.get_pitcher_elo(player_id))

    def get_batter_elo(self, player_id):
        return self.batter_elo.get(int(player_id), 1500)

    def get_historical_batter_elo(self, player_id, game_pk):
        return self.batter_elo_history.get((int(player_id), int(game_pk)), self.get_batter_elo(player_id))

    def get_form_trend(self, player_id, n=3):
        """Promedio de ultimas n performances. Negativo = mala racha."""
        form = self.recent_form.get(int(player_id), [])
        if len(form) < n: return 0
        return sum(form[-n:]) / n

class LineupAnalyzer:
    def __init__(self, elo: PlayerELO):
        self.elo = elo

    def analyze_prescan(self, h_pitcher_id, a_pitcher_id):
        """Tier 1: Solo pitchers confirmados"""
        h_elo = self.elo.get_pitcher_elo(h_pitcher_id)
        a_elo = self.elo.get_pitcher_elo(a_pitcher_id)
        h_trend = self.elo.get_form_trend(h_pitcher_id)
        a_trend = self.elo.get_form_trend(a_pitcher_id)
        return {
            'elo_starter_diff': h_elo - a_elo,
            'elo_starter_home': h_elo,
            'elo_starter_away': a_elo,
            'starter_trend_home': h_trend,
            'starter_trend_away': a_trend,
        }

    def analyze_confirmed(self, home_batters, away_batters, h_pitcher_id, a_pitcher_id):
        """Tier 2: Lineup completo confirmado"""
        h_elos = [self.elo.get_batter_elo(b['personId']) for b in home_batters[:9] if b.get('personId', 0) > 0]
        a_elos = [self.elo.get_batter_elo(b['personId']) for b in away_batters[:9] if b.get('personId', 0) > 0]

        if len(h_elos) < 7: h_elos = [1500] * 9
        if len(a_elos) < 7: a_elos = [1500] * 9

        h_elos = h_elos[:9]
        a_elos = a_elos[:9]
        while len(h_elos) < 9: h_elos.append(1450)
        while len(a_elos) < 9: a_elos.append(1450)

        h_top4 = np.mean(h_elos[:4])
        h_bot5 = np.mean(h_elos[4:])
        a_top4 = np.mean(a_elos[:4])
        a_bot5 = np.mean(a_elos[4:])

        hot_batters_h = sum(1 for b in home_batters[:9] if self.elo.get_batter_elo(b.get('personId',0)) > 1550)
        hot_batters_a = sum(1 for b in away_batters[:9] if self.elo.get_batter_elo(b.get('personId',0)) > 1550)

        starter_elo_h = self.elo.get_pitcher_elo(h_pitcher_id)
        starter_elo_a = self.elo.get_pitcher_elo(a_pitcher_id)

        return {
            'lineup_elo_home': np.mean(h_elos),
            'lineup_elo_away': np.mean(a_elos),
            'lineup_elo_diff': np.mean(h_elos) - np.mean(a_elos),
            'lineup_top4_diff': h_top4 - a_top4,
            'lineup_depth_diff': h_bot5 - a_bot5,
            'hot_batters_home': hot_batters_h,
            'hot_batters_away': hot_batters_a,
            'starter_vs_lineup_h': starter_elo_h - np.mean(a_elos),
            'starter_vs_lineup_a': starter_elo_a - np.mean(h_elos),
        }

    def compute_superiority(self, prescan, confirmed=None):
        """Formula de superioridad combinada"""
        sup = prescan.get('elo_starter_diff', 0) / 100
        if confirmed:
            sup += confirmed.get('lineup_elo_diff', 0) / 150
            sup += confirmed.get('lineup_top4_diff', 0) / 200
            sup += (confirmed.get('hot_batters_home', 0) - confirmed.get('hot_batters_away', 0)) * 0.02
        return np.clip(sup, -1, 1)

    def compute_confidence_delta(self, prescan, confirmed=None):
        """Delta para ajustar confianza del pick base"""
        sup = self.compute_superiority(prescan, confirmed)
        return sup * 0.08  # max ±8% ajuste

def create_beta_features_row(prescan, confirmed=None):
    row = {}
    row['elo_starter_diff'] = prescan.get('elo_starter_diff', 0)
    row['starter_trend_home'] = prescan.get('starter_trend_home', 0)
    row['starter_trend_away'] = prescan.get('starter_trend_away', 0)
    if confirmed:
        row['lineup_elo_diff'] = confirmed.get('lineup_elo_diff', 0)
        row['lineup_top4_diff'] = confirmed.get('lineup_top4_diff', 0)
        row['lineup_depth_diff'] = confirmed.get('lineup_depth_diff', 0)
        row['hot_batters_diff'] = confirmed.get('hot_batters_home', 0) - confirmed.get('hot_batters_away', 0)
        row['starter_vs_lineup_h'] = confirmed.get('starter_vs_lineup_h', 0)
        row['starter_vs_lineup_a'] = confirmed.get('starter_vs_lineup_a', 0)
    return row
