#!/usr/bin/env python3
"""
OMEGA PROPS v3.0 — Comprehensive Player Props Scanner
Uses: DB + Savant + XGBoost V3 specialists (H, TB, HR, R) + K V3 (pitcher)
Features: 35 total (batter L3/L10/season × 5 stats, savant batter × 6, pitcher L3/L10 × 6, pitcher savant × 5, park, home/away, chase rate)
"""
import os, sys, sqlite3, warnings, argparse, json
import numpy as np
import joblib
import statsapi
from datetime import datetime
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from sync import sync_savant_daily, sync_mlb_schedule

BASE_DIR = str(CONFIG.paths.base)
DB_PATH = str(CONFIG.paths.db)
SPEC_DIR = str(CONFIG.paths.batter_props / 'specialists')
K_MODEL = str(CONFIG.paths.k_props / 'k_xgb.pkl')
K_FEATS = str(CONFIG.paths.k_props / 'features.json')

# Colors
C_BOLD="\033[1m"; C_CYAN="\033[96m"; C_GREEN="\033[92m"; C_YELLOW="\033[93m"
C_RED="\033[91m"; C_DIM="\033[2m"; C_RESET="\033[0m"; C_PURPLE="\033[95m"
C_WHITE="\033[97m"

# Park factors (batter version — opposite of pitcher K version)
PARK_FACTORS = {
    'Colorado Rockies': 1.15, 'Cincinnati Reds': 1.05, 'Texas Rangers': 1.04,
    'Baltimore Orioles': 1.03, 'Boston Red Sox': 1.02, 'New York Yankees': 1.01,
    'Philadelphia Phillies': 1.01, 'Atlanta Braves': 1.00, 'Chicago White Sox': 1.00,
    'Milwaukee Brewers': 1.00, 'St. Louis Cardinals': 1.00, 'Toronto Blue Jays': 1.00,
    'Miami Marlins': 0.99, 'Pittsburgh Pirates': 0.99, 'Seattle Mariners': 0.99,
    'San Francisco Giants': 0.98, 'Detroit Tigers': 0.98, 'Kansas City Royals': 0.98,
    'Los Angeles Angels': 0.97, 'Tampa Bay Rays': 0.97, 'Minnesota Twins': 0.97,
    'Cleveland Guardians': 0.96, 'New York Mets': 0.96, 'Arizona Diamondbacks': 0.95,
    'Los Angeles Dodgers': 0.95, 'Houston Astros': 0.95, 'Oakland Athletics': 0.94,
    'San Diego Padres': 0.93, 'Washington Nationals': 0.93, 'Chicago Cubs': 0.92,
}

# V3 Feature names (must match training order)
FEAT_NAMES = ['l3_h','l3_r','l3_hr','l3_tb','l3_k',
              'l10_h','l10_r','l10_hr','l10_tb','l10_k',
              'szn_h','szn_r','szn_hr','szn_tb','szn_k',
              'b_xwoba','b_ev','b_barrel','b_hardhit','b_batspeed','b_sweetspot',
              'p_era','p_whip','p_k9','p_strikepct','p_l10_era','p_l10_k9',
              'p_xwoba','p_spin','p_ext','p_ev','p_barrel',
              'park','is_home','chase']


class PropsAnalyzerV3:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.batter_models = {}  # prop -> XGBClassifier
        self.k_model = None
        self._load_models()
        self._build_caches()

    def _load_models(self):
        # Batter models (H, TB, HR, R)
        for prop in ['H', 'TB', 'HR', 'R']:
            p = os.path.join(SPEC_DIR, f'omega_{prop.lower()}_v3.pkl')
            if os.path.exists(p):
                self.batter_models[prop] = joblib.load(p)
        # K V3 pitcher model
        if os.path.exists(K_MODEL) and os.path.exists(K_FEATS):
            self.k_model = joblib.load(K_MODEL)
        n_b = len(self.batter_models)
        k = 'K' if self.k_model else '—'
        print(f'{C_DIM}Models loaded: batter {n_b}/4 ({", ".join(self.batter_models.keys()) or "none"}), K: {k}{C_RESET}')

    def _build_caches(self):
        c = self.conn.cursor()
        # player_id -> most recent team_id
        bp = c.execute('SELECT player_id, team_id, date FROM batter_performances ORDER BY date DESC').fetchall()
        self.p2t = {}
        for pid, tid, d in bp:
            if pid not in self.p2t: self.p2t[pid] = tid
        # pgs by player (game-level, aggregated)
        pgs = c.execute('''
            SELECT game_pk, MAX(date), player_id,
                   SUM(hits), SUM(runs), SUM(hr), SUM(tb), SUM(k)
            FROM player_game_stats GROUP BY game_pk, player_id
        ''').fetchall()
        self.pgs_by_p = defaultdict(list)
        for r in pgs: self.pgs_by_p[r[2]].append(r)
        for pid in self.pgs_by_p: self.pgs_by_p[pid].sort(key=lambda x: x[1], reverse=True)
        # pp by player
        pp = c.execute('''SELECT game_pk, date, player_id, ip, hits, runs, er, k, bb, pitches, strikes
                         FROM pitcher_performances WHERE role="starter"''').fetchall()
        self.pp_by_p = defaultdict(list)
        for r in pp: self.pp_by_p[r[2]].append(r)
        for pid in self.pp_by_p: self.pp_by_p[pid].sort(key=lambda x: x[1], reverse=True)
        # sbd by player (batter savant)
        sbd = c.execute('SELECT player_id, game_date, avg_xwoba, avg_ev, barrels, hard_hits, avg_bat_speed, sweet_spot_count FROM savant_batter_daily').fetchall()
        self.sbd_by_p = defaultdict(list)
        for r in sbd: self.sbd_by_p[r[0]].append(r)
        for pid in self.sbd_by_p: self.sbd_by_p[pid].sort(key=lambda x: x[1])
        # spd by player (pitcher savant)
        spd = c.execute('SELECT player_id, game_date, avg_xwoba_allowed, avg_release_spin_rate, avg_release_extension, avg_ev_allowed, barrels_allowed, bbe FROM savant_pitcher_daily').fetchall()
        self.spd_by_p = defaultdict(list)
        for r in spd: self.spd_by_p[r[0]].append(r)
        for pid in self.spd_by_p: self.spd_by_p[pid].sort(key=lambda x: x[1])
        # Team ID cache
        self.team_ids = {}
        for h in c.execute('SELECT home_team, away_team, home_team_id, away_team_id FROM historico_partidos').fetchall():
            if h[0] not in self.team_ids: self.team_ids[h[0]] = h[2]
            if h[1] not in self.team_ids: self.team_ids[h[1]] = h[3]
        # Chase rate
        sb = c.execute('SELECT sb.player_id, sb.game_date, sb.o_swings, sb.o_pitches FROM savant_batter_daily sb WHERE sb.o_pitches > 0').fetchall()
        self.chase_data = defaultdict(lambda: [0, 0])
        for pid, gd, o, p in sb:
            tid = self.p2t.get(pid)
            if not tid: continue
            ym = gd[:7] if gd else '2025-01'
            self.chase_data[(tid, ym)][0] += o if o else 0
            self.chase_data[(tid, ym)][1] += p if p else 0

    def chase_rate(self, team, date):
        tid = self.team_ids.get(team)
        if not tid: return 0.30
        yr, mo = date[:4], int(date[5:7])
        total_o, total_p = 0, 0
        for m in range(1, mo+1):
            key = (tid, f'{yr}-{m:02d}')
            if key in self.chase_data:
                total_o += self.chase_data[key][0]
                total_p += self.chase_data[key][1]
        if total_p == 0:
            for m in range(1, 13):
                key = (tid, '2025-' + f'{m:02d}')
                if key in self.chase_data:
                    total_o += self.chase_data[key][0]
                    total_p += self.chase_data[key][1]
        return total_o / total_p if total_p else 0.30

    def batter_features(self, pid, opp_pitcher_id, opp_team, home_park, is_home, date):
        # Batter L3/L10/season
        pgr = self.pgs_by_p.get(pid, [])
        l3 = [g for g in pgr if g[1] < date][:3]
        l10 = [g for g in pgr if g[1] < date][:10]
        szn = [g for g in pgr if g[1] < date and str(g[1])[:4] == date[:4]]

        def avg(games, idx):
            vals = [g[idx] for g in games if g[idx] is not None]
            return np.mean(vals) if vals else 0

        b = [avg(l3, 3), avg(l3, 4), avg(l3, 5), avg(l3, 6), avg(l3, 7),
             avg(l10, 3), avg(l10, 4), avg(l10, 5), avg(l10, 6), avg(l10, 7),
             avg(szn, 3), avg(szn, 4), avg(szn, 5), avg(szn, 6), avg(szn, 7)]
        # Savant batter
        s = [r for r in self.sbd_by_p.get(pid, []) if r[1] < date][-10:]
        if s:
            bs = [np.mean([r[2] for r in s if r[2]]),
                  np.mean([r[3] for r in s if r[3]]),
                  np.mean([r[4] for r in s if r[4]]),
                  np.mean([r[5] for r in s if r[5]]),
                  np.mean([r[6] for r in s if r[6]]),
                  np.mean([r[7] for r in s if r[7]])]
        else:
            bs = [0.310, 88.0, 5.0, 30.0, 70.0, 30.0]
        # Pitcher
        pgr_p = self.pp_by_p.get(opp_pitcher_id, [])
        p_l3 = [g for g in pgr_p if g[1] < date][:3]
        p_l10 = [g for g in pgr_p if g[1] < date][:10]
        p_l3_ip = sum(r[3] for r in p_l3 if r[3])
        p_l3_er = sum(r[5] for r in p_l3 if r[5])
        p_l3_k = sum(r[7] for r in p_l3 if r[7])
        p_l3_bb = sum(r[8] for r in p_l3 if r[8])
        p_l3_h = sum(r[4] for r in p_l3 if r[4])
        p_l3_p = sum(r[9] for r in p_l3 if r[9])
        p_l3_s = sum(r[10] for r in p_l3 if r[10])
        p_era = (p_l3_er * 9 / p_l3_ip) if p_l3_ip > 0 else 4.5
        p_whip = (p_l3_h + p_l3_bb) / p_l3_ip if p_l3_ip > 0 else 1.3
        p_k9 = (p_l3_k * 9 / p_l3_ip) if p_l3_ip > 0 else 7.5
        p_strike_pct = p_l3_s / p_l3_p if p_l3_p > 0 else 0.62
        p_l10_ip = sum(r[3] for r in p_l10 if r[3])
        p_l10_er = sum(r[5] for r in p_l10 if r[5])
        p_l10_k = sum(r[7] for r in p_l10 if r[7])
        p_l10_era = (p_l10_er * 9 / p_l10_ip) if p_l10_ip > 0 else 4.5
        p_l10_k9 = (p_l10_k * 9 / p_l10_ip) if p_l10_ip > 0 else 7.5
        # Pitcher savant
        ps = [r for r in self.spd_by_p.get(opp_pitcher_id, []) if r[1] < date]
        ps = [r for r in ps if r[7] and r[7] >= 3][-10:]
        if ps:
            tbb = sum(r[7] for r in ps)
            psv = [np.mean([r[2] for r in ps if r[2]]),
                   np.mean([r[3] for r in ps if r[3]]),
                   np.mean([r[4] for r in ps if r[4]]),
                   np.mean([r[5] for r in ps if r[5]]),
                   sum(r[6] for r in ps) / tbb if tbb else 0.06]
        else:
            psv = [0.320, 2200, 6.0, 89.0, 0.06]
        ch = self.chase_rate(opp_team, date)
        park_factor = PARK_FACTORS.get(home_park, 1.0)
        return b + bs + [p_era, p_whip, p_k9, p_strike_pct, p_l10_era, p_l10_k9] + psv + [park_factor, float(is_home), ch]

    def predict_batter_props(self, pid, opp_pitcher_id, opp_team, home_park, is_home, date):
        X = np.array([self.batter_features(pid, opp_pitcher_id, opp_team, home_park, is_home, date)])
        X = np.nan_to_num(X, nan=0.0)
        probs = {}
        for prop, m in self.batter_models.items():
            try:
                probs[prop] = float(m.predict_proba(X)[0][1])
            except Exception as e:
                probs[prop] = None
        return probs

    def predict_pitcher_k(self, name, opp_team, home_park, date):
        """Use K V3 model (XGBoost) for pitcher strikeouts."""
        if not self.k_model: return None
        pid_row = self.conn.execute('SELECT player_id FROM pitcher_performances WHERE player_name=? OR player_name LIKE ? ORDER BY date DESC LIMIT 1',
                                    (name, f'%{name}%')).fetchone()
        if not pid_row: return None
        pid = pid_row[0]
        # Build features (same as K V3)
        l3 = self.conn.execute('SELECT k, ip, pitches, strikes FROM pitcher_performances WHERE player_id=? AND role="starter" AND date<? ORDER BY date DESC LIMIT 3', (pid, date)).fetchall()
        if not l3: return None
        l3_k = sum(r[0] for r in l3 if r[0])
        l3_ip = sum(r[1] for r in l3 if r[1])
        l3_k9 = l3_k * 9 / l3_ip if l3_ip > 0 else 0
        l3_ip_avg = l3_ip / len(l3)
        l3_p = sum(r[2] for r in l3 if r[2])
        l3_s = sum(r[3] for r in l3 if r[3])
        sp = l3_s / l3_p if l3_p else 0.62
        l10 = self.conn.execute('SELECT k, ip FROM pitcher_performances WHERE player_id=? AND role="starter" AND date<? ORDER BY date DESC LIMIT 10', (pid, date)).fetchall()
        l10_k = sum(r[0] for r in l10 if r[0])
        l10_ip = sum(r[1] for r in l10 if r[1])
        l10_k9 = l10_k * 9 / l10_ip if l10_ip > 0 else 0
        yr = date[:4]
        szn = self.conn.execute('SELECT SUM(k), SUM(ip) FROM pitcher_performances WHERE player_id=? AND role="starter" AND date LIKE ? AND date<?', (pid, f'{yr}%', date)).fetchall()[0]
        szn_k9 = szn[0] * 9 / szn[1] if szn[1] else 0
        car = self.conn.execute('SELECT SUM(k), SUM(ip) FROM pitcher_performances WHERE player_id=? AND role="starter" AND date<?', (pid, date)).fetchall()[0]
        car_k9 = car[0] * 9 / car[1] if car[1] else 0
        sav = self.conn.execute('SELECT avg_xwoba_allowed, avg_release_spin_rate, avg_release_extension, avg_ev_allowed, barrels_allowed, bbe FROM savant_pitcher_daily WHERE player_id=? AND bbe>=3 AND game_date<? ORDER BY game_date DESC LIMIT 10', (pid, date)).fetchall()
        if sav and len(sav) >= 3:
            sxw = np.mean([x[0] for x in sav if x[0]])
            ssp = np.mean([x[1] for x in sav if x[1]])
            sex = np.mean([x[2] for x in sav if x[2]])
            sev = np.mean([x[3] for x in sav if x[3]])
            tb = sum(x[5] for x in sav)
            bbr = sum(x[4] for x in sav) / tb if tb else 0.06
        else:
            sxw, ssp, sex, sev, bbr = 0.32, 2200, 6.0, 89, 0.06
        ch = self.chase_rate(opp_team, date)
        park = PARK_FACTORS.get(home_park, 1.0)
        last_g = self.conn.execute('SELECT date FROM pitcher_performances WHERE player_id=? AND role="starter" AND date<? ORDER BY date DESC LIMIT 1', (pid, date)).fetchall()
        from datetime import datetime as dt
        dr = 5
        if last_g:
            try:
                d1 = dt.strptime(str(date)[:10], '%Y-%m-%d')
                d2 = dt.strptime(str(last_g[0][0])[:10], '%Y-%m-%d')
                dr = (d1 - d2).days
            except: pass
        feats = [l3_k9, l10_k9, szn_k9, car_k9, sxw, ssp, sex, sev, bbr, sp, l3_ip_avg, ch, park, dr]
        return max(0, float(self.k_model.predict(np.array([feats]))[0]))

    def scan(self, date_str):
        sched = statsapi.schedule(date=date_str)
        valid = {'Scheduled', 'Pre-Game', 'Warmup', 'In Progress', 'Final', 'Completed'}
        games = [g for g in sched if g.get('status') in valid]

        results = defaultdict(list)

        for g in games:
            hn, an = g.get('home_name'), g.get('away_name')
            hp, ap = g.get('home_probable_pitcher', ''), g.get('away_probable_pitcher', '')
            hp_id, ap_id = g.get('home_probable_pitcher_id', 0), g.get('away_probable_pitcher_id', 0)
            park = PARK_FACTORS.get(hn, 1.0)

            # K predictions (pitchers)
            if self.k_model:
                if hp:
                    kp = self.predict_pitcher_k(hp, an, hn, date_str)
                    if kp: results['K (Pitcher)'].append({'pitcher': hp, 'team': hn, 'opp': an, 'proj_k': kp})
                if ap:
                    kp = self.predict_pitcher_k(ap, hn, hn, date_str)
                    if kp: results['K (Pitcher)'].append({'pitcher': ap, 'team': an, 'opp': hn, 'proj_k': kp})

            # Get hitters from DB for each team (most recent regulars in lineup positions)
            # Use lineups from boxscore if game is in progress, else use predicted from recent
            # For simplicity, predict top 9 hitters from each team's recent boxscore
            for side, team, opp, opp_pid, opp_pname, opp_id in [
                ('home', hn, an, ap_id, ap, ap_id),
                ('away', an, hn, hp_id, hp, hp_id)
            ]:
                if not opp_id:
                    # Try to get pitcher id by name
                    r = self.conn.execute('SELECT player_id FROM pitcher_performances WHERE player_name=? OR player_name LIKE ? ORDER BY date DESC LIMIT 1',
                                          (opp_pname, f'%{opp_pname}%')).fetchall()
                    if r: opp_id = r[0][0]
                if not opp_id: continue
                # Get top hitters from team's active or projected lineup (prevents predicting inactive/resting players)
                team_id = self.team_ids.get(team)
                if not team_id: continue
                
                hitters = []
                # 1. Boxscore actual starting lineup (if warmups/pre-game has lineups)
                try:
                    box = statsapi.boxscore_data(g['gamePk'])
                    batters = box.get(f'{side}Batters', [])
                    starters = [b for b in batters if b.get('personId', 0) > 0 and not b.get('substitution', False)]
                    if len(starters) >= 8:
                        hitters = [b['personId'] for b in starters[:9]]
                except:
                    pass

                # 2. Markov LineupPredictor projection fallback
                if not hitters:
                    try:
                        from sync import LineupPredictor
                        predictor = LineupPredictor()
                        pred_lineup = predictor.predict_lineup(team_id, date_str)
                        if pred_lineup and len(pred_lineup) >= 8:
                            hitters = [b['personId'] for b in pred_lineup[:9]]
                    except:
                        pass

                # 3. Last resort fallback: top 9 regulars from recent database performances
                if not hitters:
                    try:
                        rows = self.conn.execute('''
                            SELECT DISTINCT player_id FROM batter_performances
                            WHERE team_id = ? ORDER BY date DESC LIMIT 12
                        ''', (team_id,)).fetchall()
                        hitters = [r[0] for r in rows[:9]]
                    except:
                        pass

                if not hitters: continue
                for pid in hitters:
                    if pid not in self.pgs_by_p: continue
                    probs = self.predict_batter_props(pid, opp_id, opp, hn, side == 'home', date_str)
                    for prop, prob in probs.items():
                        if prob is None: continue
                        # Add park/savant bonus
                        if prop == 'H' and prob > 0.55:
                            results['HITS (1+)'].append({'player_id': pid, 'team': team, 'match': f'{an} @ {hn}', 'prob': prob})
                        elif prop == 'TB' and prob > 0.40:
                            results['TB (1.5+)'].append({'player_id': pid, 'team': team, 'match': f'{an} @ {hn}', 'prob': prob})
                        elif prop == 'HR' and prob > 0.10:
                            results['HR (1+)'].append({'player_id': pid, 'team': team, 'match': f'{an} @ {hn}', 'prob': prob})
                        elif prop == 'R' and prob > 0.45:
                            results['RUNS (1+)'].append({'player_id': pid, 'team': team, 'match': f'{an} @ {hn}', 'prob': prob})
        return results


def get_player_name(conn, pid):
    r = conn.execute('SELECT player_name FROM batter_performances WHERE player_id=? ORDER BY date DESC LIMIT 1', (pid,)).fetchone()
    if not r:
        r = conn.execute('SELECT player_name FROM pitcher_performances WHERE player_id=? ORDER BY date DESC LIMIT 1', (pid,)).fetchone()
    return r[0] if r else f'Player {pid}'


def scan_props(date_str=None, skip_sync=False, skip_train=False):
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
        
    # 1. Update data
    if not skip_sync:
        print("📡 Sincronizando calendario y datos de Savant...")
        try:
            sync_mlb_schedule(days_back=3)
            sync_savant_daily()
        except Exception as e:
            print(f"⚠️ Error en la sincronización: {e}")
            
    # 2. Retrain models
    if not skip_train:
        try:
            from models.train_props import run_retraining
            run_retraining()
        except Exception as e:
            print(f"⚠️ Error al reentrenar modelos: {e}")

    print(f"\n{C_BOLD}{C_PURPLE}{'='*120}{C_RESET}")
    print(f"{C_BOLD}{C_PURPLE} OMEGA PROPS v3.0 — DB + Savant + XGBoost (K V3 + H/TB/HR/R) — {date_str}{C_RESET}")
    print(f"{C_BOLD}{C_PURPLE}{'='*120}{C_RESET}")

    analyzer = PropsAnalyzerV3()
    results = analyzer.scan(date_str)
    conn = analyzer.conn  # keep open for name lookups

    print(f"\n{C_BOLD}{C_CYAN}{'='*120}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN} OMEGA PROPS — RESULTS {date_str}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}{'='*120}{C_RESET}\n")

    for title, data in sorted(results.items()):
        if title == 'K (Pitcher)':
            data.sort(key=lambda x: x['proj_k'], reverse=True)
        else:
            data.sort(key=lambda x: x['prob'], reverse=True)

        print(f"{C_BOLD}{C_CYAN}  {title}{C_RESET}")
        for r in data[:10]:
            if title == 'K (Pitcher)':
                kp = r['proj_k']
                color = C_GREEN if kp >= 7 else (C_YELLOW if kp >= 5.5 else C_WHITE)
                print(f"    {color}{kp:.1f}K{C_RESET} {r['pitcher'][:22]:<23} {r['team'][:16]:<17} vs {r['opp'][:18]:<18}")
            else:
                p = r['prob']
                color = C_GREEN if p >= 0.65 else (C_YELLOW if p >= 0.55 else C_WHITE)
                name = get_player_name(conn, r['player_id'])
                print(f"    {color}{p:.1%}{C_RESET} {name[:22]:<23} {r['team'][:16]:<17} {r['match'][:30]:<30}")
        print()
    print(f"{'='*120}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OMEGA PROPS v3.0')
    parser.add_argument('--date', type=str, default=None)
    args = parser.parse_args()
    scan_props(args.date)
