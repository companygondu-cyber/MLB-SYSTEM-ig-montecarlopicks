"""omega_k_analysis.py — K (strikeout) analysis V3 (REVERSIBLE / TEST ONLY)
Uses XGBoost model trained on 5406 games (2025-2026) with 14 features:
  K/9 L3, K/9 L10, K/9 season, K/9 career, xwOBA, spin, ext, EV,
  barrel rate, strike%, IP avg, opp chase rate, park K, days rest
Pulls LIVE probable pitchers from MLB StatsAPI
Falls back to V1 formula if model is unavailable"""
import sqlite3, numpy as np, joblib, json
from math import exp, factorial
from pathlib import Path
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
import statsapi

DB_PATH = CONFIG.paths.db
MODEL_PATH = CONFIG.paths.k_props / 'k_xgb.pkl'
FEAT_PATH = CONFIG.paths.k_props / 'features.json'

PARK_K_FACTORS = {
    'Colorado Rockies': 0.92, 'Boston Red Sox': 0.95, 'Cincinnati Reds': 0.96,
    'Texas Rangers': 0.97, 'Baltimore Orioles': 0.97, 'Philadelphia Phillies': 0.97,
    'New York Yankees': 0.98, 'Minnesota Twins': 0.98, 'Chicago Cubs': 0.98,
    'Atlanta Braves': 0.99, 'Houston Astros': 0.99, 'Arizona Diamondbacks': 0.99,
    'Milwaukee Brewers': 1.00, 'Pittsburgh Pirates': 1.00, 'Cleveland Guardians': 1.00,
    'Kansas City Royals': 1.00, 'Detroit Tigers': 1.00, 'Toronto Blue Jays': 1.01,
    'St. Louis Cardinals': 1.01, 'Chicago White Sox': 1.01, 'New York Mets': 1.01,
    'Tampa Bay Rays': 1.02, 'Miami Marlins': 1.02, 'Oakland Athletics': 1.02,
    'Los Angeles Angels': 1.03, 'Washington Nationals': 1.03, 'Seattle Mariners': 1.04,
    'San Diego Padres': 1.04, 'Los Angeles Dodgers': 1.05, 'San Francisco Giants': 1.05,
}


def poisson_over(k_proj, line):
    if k_proj <= 0: return 0
    k = int(line)
    p_under = sum((k_proj**i * exp(-k_proj)) / factorial(i) for i in range(k+1))
    return 1 - p_under


def get_pitcher_id_by_name(conn, name):
    if not name: return 0
    rows = conn.execute('''
        SELECT player_id FROM pitcher_performances
        WHERE player_name = ? OR player_name LIKE ?
        ORDER BY date DESC LIMIT 1
    ''', (name, f'%{name}%')).fetchall()
    return rows[0][0] if rows else 0


class KAnalyzer:
    def __init__(self, conn):
        self.conn = conn
        self.model = None
        self.feat_names = None
        if MODEL_PATH.exists() and FEAT_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                self.feat_names = json.load(open(FEAT_PATH))
            except Exception as e:
                print(f'Model load failed: {e}')

        # Pre-compute team_id cache
        self.team_ids = {}
        for tn in conn.execute('SELECT DISTINCT home_team FROM historico_partidos UNION SELECT DISTINCT away_team FROM historico_partidos').fetchall():
            r = conn.execute('SELECT home_team_id FROM historico_partidos WHERE home_team=? OR away_team=? LIMIT 1', (tn[0], tn[0])).fetchall()
            self.team_ids[tn[0]] = r[0][0] if r else None

        # Pre-compute chase rate per team per month
        # Build in-memory player_id -> most recent team_id (avoids slow JOIN)
        bp_rows = conn.execute('SELECT player_id, team_id, date FROM batter_performances ORDER BY date DESC').fetchall()
        p2t = {}
        for pid, tid, d in bp_rows:
            if pid not in p2t: p2t[pid] = tid
        sb_rows = conn.execute('SELECT player_id, game_date, o_swings, o_pitches FROM savant_batter_daily WHERE o_pitches > 0').fetchall()
        self._chase_data = defaultdict(lambda: [0, 0])
        for pid, gd, o, p in sb_rows:
            tid = p2t.get(pid)
            if not tid: continue
            ym = gd[:7] if gd else '2025-01'
            self._chase_data[(tid, ym)][0] += o if o else 0
            self._chase_data[(tid, ym)][1] += p if p else 0

    def chase_rate(self, team, date):
        tid = self.team_ids.get(team)
        if not tid: return 0.30
        yr, mo = date[:4], int(date[5:7])
        total_o, total_p = 0, 0
        for m in range(1, mo+1):
            key = (tid, f'{yr}-{m:02d}')
            if key in self._chase_data:
                total_o += self._chase_data[key][0]
                total_p += self._chase_data[key][1]
        if total_p == 0:
            for m in range(1, 13):
                key = (tid, '2025-' + f'{m:02d}')
                if key in self._chase_data:
                    total_o += self._chase_data[key][0]
                    total_p += self._chase_data[key][1]
        return total_o / total_p if total_p else 0.30

    def get_features(self, pid, opp_team, home_park, date):
        from datetime import datetime
        # L3
        l3 = self.conn.execute('''
            SELECT k, ip, pitches, strikes FROM pitcher_performances
            WHERE player_id=? AND role='starter' AND date<?
            ORDER BY date DESC LIMIT 3
        ''', (pid, date)).fetchall()
        if not l3: return None
        l3_k = sum(r[0] for r in l3 if r[0])
        l3_ip = sum(r[1] for r in l3 if r[1])
        l3_k9 = l3_k * 9 / l3_ip if l3_ip > 0 else 0
        l3_ip_avg = l3_ip / len(l3)
        l3_p = sum(r[2] for r in l3 if r[2])
        l3_s = sum(r[3] for r in l3 if r[3])
        sp = l3_s / l3_p if l3_p else 0.62

        l10 = self.conn.execute('''
            SELECT k, ip FROM pitcher_performances
            WHERE player_id=? AND role='starter' AND date<?
            ORDER BY date DESC LIMIT 10
        ''', (pid, date)).fetchall()
        l10_k = sum(r[0] for r in l10 if r[0])
        l10_ip = sum(r[1] for r in l10 if r[1])
        l10_k9 = l10_k * 9 / l10_ip if l10_ip > 0 else 0

        yr = date[:4]
        szn = self.conn.execute('''
            SELECT SUM(k), SUM(ip) FROM pitcher_performances
            WHERE player_id=? AND role='starter' AND date LIKE ? AND date<?
        ''', (pid, f'{yr}%', date)).fetchall()[0]
        szn_k9 = szn[0] * 9 / szn[1] if szn[1] else 0

        car = self.conn.execute('''
            SELECT SUM(k), SUM(ip) FROM pitcher_performances
            WHERE player_id=? AND role='starter' AND date<?
        ''', (pid, date)).fetchall()[0]
        car_k9 = car[0] * 9 / car[1] if car[1] else 0

        sav = self.conn.execute('''
            SELECT avg_xwoba_allowed, avg_release_spin_rate, avg_release_extension,
                   avg_ev_allowed, barrels_allowed, bbe
            FROM savant_pitcher_daily
            WHERE player_id=? AND bbe>=3 AND game_date<?
            ORDER BY game_date DESC LIMIT 10
        ''', (pid, date)).fetchall()
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
        park = PARK_K_FACTORS.get(home_park, 1.0)

        last_g = self.conn.execute('''
            SELECT date FROM pitcher_performances
            WHERE player_id=? AND role='starter' AND date<?
            ORDER BY date DESC LIMIT 1
        ''', (pid, date)).fetchall()
        dr = 5
        if last_g:
            try:
                d1 = datetime.strptime(str(date)[:10], '%Y-%m-%d')
                d2 = datetime.strptime(str(last_g[0][0])[:10], '%Y-%m-%d')
                dr = (d1 - d2).days
            except: pass

        return [l3_k9, l10_k9, szn_k9, car_k9, sxw, ssp, sex, sev, bbr, sp, l3_ip_avg, ch, park, dr], {
            'k9_l3': l3_k9, 'k9_l10': l10_k9, 'k9_szn': szn_k9, 'k9_car': car_k9,
            'xwoba': sxw, 'spin': ssp, 'ext': sex, 'ev': sev, 'barrel': bbr,
            'strike_pct': sp, 'ip_avg': l3_ip_avg, 'chase': ch, 'park': park, 'rest': dr
        }

    def analyze_pitcher(self, name, opp_team, home_park, date='2026-06-03'):
        pid = get_pitcher_id_by_name(self.conn, name)
        if not pid: return None

        result = self.get_features(pid, opp_team, home_park, date)
        if not result: return None
        X, info = result

        if self.model is not None:
            k_proj = float(self.model.predict(np.array([X]))[0])
            k_proj = max(0, k_proj)
            method = 'XGBoost-V3'
        else:
            k_proj = self._v1_fallback(X, info)
            method = 'Formula-V1'

        return {
            'name': name,
            'k9_l3': info['k9_l3'],
            'k9_szn': info['k9_szn'],
            'xwoba': info['xwoba'],
            'spin': info['spin'],
            'ext': info['ext'],
            'ip_l3': info['ip_avg'],
            'chase': info['chase'],
            'park_k': info['park'],
            'rest': info['rest'],
            'proj_k': k_proj,
            'method': method
        }

    def _v1_fallback(self, X, info):
        base_k9 = info['k9_l3'] * 0.6 + info['k9_szn'] * 0.4
        spin_bonus = 1.08 if info['spin'] > 2400 else (1.04 if info['spin'] > 2300 else (0.94 if info['spin'] < 2100 else 1.0))
        ext_bonus = 1.05 if info['ext'] > 6.5 else (0.95 if info['ext'] < 5.8 else 1.0)
        dom_adj = 1.08 if info['xwoba'] < 0.280 else (1.04 if info['xwoba'] < 0.310 else (0.92 if info['xwoba'] > 0.360 else (0.96 if info['xwoba'] > 0.340 else 1.0)))
        workload_adj = 0.95 if info['ip_avg'] > 6.5 else (0.90 if info['ip_avg'] < 4.5 else 1.0)
        return (base_k9 * spin_bonus * ext_bonus * dom_adj * info['park'] * workload_adj) * (info['ip_avg'] / 9)


def analyze_date(date_str):
    conn = sqlite3.connect(str(DB_PATH))
    analyzer = KAnalyzer(conn)

    sched = statsapi.schedule(date=date_str)
    valid = {'Scheduled', 'Pre-Game', 'Warmup', 'In Progress', 'Final', 'Completed'}
    games = [g for g in sched if g.get('status') in valid]

    results = []
    seen = set()
    for g in games:
        hn, an = g.get('home_name'), g.get('away_name')
        hp, ap = g.get('home_probable_pitcher', ''), g.get('away_probable_pitcher', '')

        if hp and hp not in seen:
            seen.add(hp)
            stats = analyzer.analyze_pitcher(hp, an, hn, date_str)
            if stats:
                stats.update({'team': hn, 'opp': an, 'side': 'home'})
                results.append(stats)
        if ap and ap not in seen:
            seen.add(ap)
            stats = analyzer.analyze_pitcher(ap, hn, hn, date_str)
            if stats:
                stats.update({'team': an, 'opp': hn, 'side': 'away'})
                results.append(stats)

    conn.close()
    return results


if __name__ == '__main__':
    import sys
    date_str = sys.argv[1] if len(sys.argv) > 1 else '2026-06-03'

    results = analyze_date(date_str)

    for r in results:
        best_over = None
        for line_val in [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
            p_over = poisson_over(r['proj_k'], line_val)
            if p_over >= 0.55:
                best_over = (f'o{line_val}', p_over)
                break
        r['over'] = best_over

        best_under = None
        for line_val in [3.5, 4.5, 5.5, 6.5, 7.5]:
            p_under = 1 - poisson_over(r['proj_k'], line_val)
            if p_under >= 0.60 and p_under <= 0.85:
                best_under = (f'u{line_val}', p_under)
                break
        r['under'] = best_under

    overs = sorted([r for r in results if r['over']], key=lambda x: x['over'][1], reverse=True)
    unders = sorted([r for r in results if r['under'] and not r['over']], key=lambda x: x['under'][1], reverse=True)
    no_play = [r for r in results if not r['over'] and not r['under']]

    method = results[0]['method'] if results else 'N/A'

    print('=' * 130)
    print(f'  K PROPS ANALYSIS V3 ({method}) — {date_str} (LIVE MLB Probable Pitchers)')
    print('=' * 130)
    print()

    print('=' * 130)
    print(f'  OVER PICKS ({len(overs)} pitchers)')
    print('=' * 130)
    print(f'{"Pitcher":22s} {"Team":18s} {"Opp":18s} {"K/9_L3":>7s} {"xwOBA":>6s} {"Spin":>5s} {"Ext":>5s} {"Chase":>6s} {"ParkK":>6s} {"Rest":>5s} {"ProjK":>6s} {"Line":>6s} {"Prob":>6s}')
    print('-' * 130)
    for r in overs:
        line, prob = r['over']
        print(f'{r["name"]:22s} {r["team"][:18]:18s} {r["opp"][:18]:18s} {r["k9_l3"]:7.2f} {r["xwoba"]:6.3f} {r["spin"]:5.0f} {r["ext"]:5.2f} {r["chase"]:6.1%} {r["park_k"]:6.2f} {r["rest"]:5d} {r["proj_k"]:6.1f} {line:>6s} {prob:6.1%}')

    print()
    print('=' * 130)
    print(f'  UNDER PICKS ({len(unders)} pitchers)')
    print('=' * 130)
    print(f'{"Pitcher":22s} {"Team":18s} {"Opp":18s} {"K/9_L3":>7s} {"xwOBA":>6s} {"Spin":>5s} {"Ext":>5s} {"Chase":>6s} {"ParkK":>6s} {"Rest":>5s} {"ProjK":>6s} {"Line":>6s} {"Prob":>6s}')
    print('-' * 130)
    for r in unders:
        line, prob = r['under']
        print(f'{r["name"]:22s} {r["team"][:18]:18s} {r["opp"][:18]:18s} {r["k9_l3"]:7.2f} {r["xwoba"]:6.3f} {r["spin"]:5.0f} {r["ext"]:5.2f} {r["chase"]:6.1%} {r["park_k"]:6.2f} {r["rest"]:5d} {r["proj_k"]:6.1f} {line:>6s} {prob:6.1%}')

    if no_play:
        print()
        print('=' * 130)
        print(f'  NO PLAY ({len(no_play)} pitchers - low confidence)')
        print('=' * 130)
        for r in no_play:
            print(f'  {r["name"]:22s} {r["team"][:18]:18s} K/9_L3={r["k9_l3"]:.2f}, ProjK={r["proj_k"]:.1f}')
