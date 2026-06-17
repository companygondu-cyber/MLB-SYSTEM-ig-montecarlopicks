"""Lineup Predictor — reversible Markov model for pre-scan."""
import sqlite3, os, warnings, logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import concurrent.futures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'omega_2026_BETA.db')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

NAME_TO_TID = {
    "Arizona Diamondbacks":109,"Atlanta Braves":144,"Baltimore Orioles":110,
    "Boston Red Sox":111,"Chicago Cubs":112,"Chicago White Sox":145,
    "Cincinnati Reds":113,"Cleveland Guardians":114,"Colorado Rockies":115,
    "Detroit Tigers":116,"Houston Astros":117,"Kansas City Royals":118,
    "Los Angeles Angels":108,"Los Angeles Dodgers":119,"Miami Marlins":146,
    "Milwaukee Brewers":158,"Minnesota Twins":142,"New York Mets":121,
    "New York Yankees":147,"Oakland Athletics":133,"Athletics":133,
    "Philadelphia Phillies":143,"Pittsburgh Pirates":134,"San Diego Padres":135,
    "San Francisco Giants":137,"Seattle Mariners":136,"St. Louis Cardinals":138,
    "Tampa Bay Rays":139,"Texas Rangers":140,"Toronto Blue Jays":141,
    "Washington Nationals":120,
}

class LineupPredictor:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
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
                import requests
                url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=injury"
                resp = requests.get(url, timeout=2.0)
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
            recent_games = past[-9:] # fallback
            
        for gdate, lineup in recent_games:
            g_obj = datetime.strptime(gdate[:10], '%Y-%m-%d')
            diff_days = (date_obj - g_obj).days
            if diff_days <= 3: weight = 3
            elif diff_days <= 7: weight = 2
            else: weight = 1
            
            seen = set()
            for slot, pid, _ in lineup:
                if pid not in seen:
                    if pid not in injured_ids:  # Exclude players on IL
                        player_apps[pid] += weight
                        seen.add(pid)
                player_slots[pid][slot] += weight

        N_window = len(recent_games)

        # Rank by appearance frequency, take top 9
        player_rank = sorted(player_apps.items(), key=lambda x: (-x[1], x[0]))
        top_9_pids = [pid for pid, _ in player_rank[:9]]
        top_9_set = set(top_9_pids)

        # Assign slots using iterative refinement:
        # 1. Start with last game's slots for players in BOTH top 9 and last game
        # 2. Fill remaining with best-fit players

        assigned = {}
        used_pids = set()

        # Phase 1: Keep last game slots for returning starters
        for slot in range(1, 10):
            pid = last_map.get(slot)
            if pid and pid in top_9_set:
                assigned[slot] = pid
                used_pids.add(pid)

        # Phase 2: For remaining players (new to top 9, or moved)
        remaining_pids = [pid for pid in top_9_pids if pid not in used_pids]
        empty_slots = [s for s in range(1, 10) if s not in assigned]

        # Match remaining players to empty slots using their modal slot
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
                # Pick the best empty slot for this player
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
        self._enrich_stats(result, date_str)
        return result

    def _enrich_stats(self, batters, date_str):
        pids = [b['personId'] for b in batters if b.get('personId', 0) > 0]
        if not pids:
            return
        fresh = {}
        stale = {}
        for pid in pids:
            cached = self._stats_cache.get(pid)
            if cached and cached.get('_ts') and (datetime.now() - cached['_ts']).total_seconds() < 3600:
                fresh[pid] = cached
            else:
                stale[pid] = True
        if stale:
            try:
                import statsapi
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
                    fut_map = {exe.submit(self._fetch_player_stats, pid, statsapi): pid for pid in stale}
                    for fut in concurrent.futures.as_completed(fut_map):
                        pid = fut_map[fut]
                        try:
                            result = fut.result()
                            if result:
                                result['_ts'] = datetime.now()
                                self._stats_cache[pid] = result
                                fresh[pid] = result
                        except:
                            pass
            except:
                pass
        for pid in stale:
            if pid not in fresh:
                db_stats = self._get_db_batter_stats(pid, date_str)
                if db_stats:
                    db_stats['_ts'] = datetime.now()
                    self._stats_cache[pid] = db_stats
                    fresh[pid] = db_stats
        for b in batters:
            pid = b.get('personId', 0)
            s = fresh.get(pid) or self._stats_cache.get(pid)
            if s:
                b['avg'] = s.get('avg', 0.250)
                b['ops'] = s.get('ops', 0.700)
                b['obp'] = s.get('obp', 0.320)
                b['slg'] = s.get('slg', 0.400)
            else:
                b['avg'] = 0.250
                b['ops'] = 0.700
                b['obp'] = 0.320
                b['slg'] = 0.400

    def _fetch_player_stats(self, pid, statsapi):
        try:
            return None # Disable HTTP fetch in predictor to avoid timeouts
            d = statsapi.player_stat_data(personId=int(pid), group="[hitting]", type="season")
            s = d.get('stats', [{}])[0].get('stats', {}) if d else {}
            if s and s.get('avg'):
                return {'avg': float(s['avg']), 'ops': float(s.get('ops', 0.700)),
                        'obp': float(s.get('obp', 0.320)), 'slg': float(s.get('slg', 0.400))}
        except:
            pass
        return None

    def _get_db_batter_stats(self, pid, date_str):
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute("""
                SELECT h_batter_1_avg, h_batter_1_ops FROM historico_partidos
                WHERE date < ? AND date >= ? AND h_batter_1_avg IS NOT NULL
            """, (date_str, f"{date_str[:4]}-01-01")).fetchall()
            conn.close()
            avgs, opss = [], []
            for row in rows:
                if row[0] and row[1]:
                    avgs.append(float(row[0]))
                    opss.append(float(row[1]))
            if avgs:
                return {'avg': sum(avgs)/len(avgs), 'ops': sum(opss)/len(opss), 'obp': 0.320, 'slg': 0.400}
        except:
            pass
        return None

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


if __name__ == '__main__':
    import sys
    lp = LineupPredictor()
    start = sys.argv[1] if len(sys.argv) > 1 else '2026-05-01'
    end = sys.argv[2] if len(sys.argv) > 2 else '2026-05-24'
    print(f"Backtesting {start} → {end}...")
    r = lp.backtest(start, end)
    print(f"  Games tested: {r['games_tested']}")
    print(f"  Player accuracy: {r['who_accuracy']:.1%} ({r['correct_who']}/{r['total_who']})")
    print(f"  Slot accuracy: {r['slot_accuracy']:.1%} ({r['correct_slots']}/{r['total_slots']})")
    print(f"  Exact 9/9 lineups: {r['exact_9']}/{r['games_with_prediction']}")
