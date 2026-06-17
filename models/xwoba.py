"""
omega_xwoba.py — Standalone xwOBA Prediction Module (REVERSIBLE / TEST ONLY)
============================================================================
Predicts expected xwOBA for each batter in today's games.

Approach:
  1. Pitcher suppression: pitcher's rolling xwOBA allowed (EWM-15)
  2. Batter form: batter's rolling xwOBA (EWM-15)
  3. Platoon split: batter vs L/R pitcher adjustment
  4. Park factor: Coors up, PETCO down, etc.
  5. Blended prediction: weighted average

Usage:
  python3 omega_xwoba.py --date 2026-06-03
  python3 omega_xwoba.py --mode test   # quick validation on historical games

To REMOVE: just delete this file and remove the import in omega_v3.py
"""

import sqlite3, argparse, warnings, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

DB_PATH = CONFIG.paths.db

# Park factors (source: FanGraphs 2024-2025 avg)
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

MLB_AVG_XWOBA = 0.310


def get_pitcher_xwoba_rolling(conn, pitcher_id, before_date, span=15):
    """Get pitcher's rolling xwOBA allowed (EWM-15)."""
    rows = conn.execute('''
        SELECT game_date, avg_xwoba_allowed, bbe
        FROM savant_pitcher_daily
        WHERE player_id = ? AND game_date < ?
        ORDER BY game_date DESC
    ''', (int(pitcher_id), before_date)).fetchall()

    if not rows:
        return MLB_AVG_XWOBA, 0

    # Filter bbe >= 3 for stability
    filtered = [(d, x, b) for d, x, b in rows if b >= 3 and x is not None]
    if len(filtered) < 2:
        return rows[0][1] if rows[0][1] else MLB_AVG_XWOBA, len(filtered)

    dates, xwobas, bbes = zip(*filtered[:span])
    # Exponential decay weights (most recent = 1.0)
    weights = [0.7 ** i for i in range(len(xwobas))]
    bbe_weights = [b / max(bbes[0], 1) for b in bbes]
    combined = [w * bw for w, bw in zip(weights, bbe_weights)]
    total = sum(combined)
    if total > 0:
        rolling = sum(x * cw for x, cw in zip(xwobas, combined)) / total
    else:
        rolling = np.mean(xwobas)

    return rolling, len(filtered)


def get_batter_xwoba_rolling(conn, batter_id, before_date, p_throws='R', span=15):
    """Get batter's rolling xwOBA (EWM-15), optionally vs L/R."""
    rows = conn.execute('''
        SELECT game_date, avg_xwoba, bbe, p_throws
        FROM savant_batter_daily
        WHERE player_id = ? AND game_date < ?
        ORDER BY game_date DESC
    ''', (int(batter_id), before_date)).fetchall()

    if not rows:
        return MLB_AVG_XWOBA, 0, 0

    # Split by handedness if available
    vs_hand = [(d, x, b) for d, x, b, h in rows if h == p_throws and b >= 2 and x is not None]
    all_valid = [(d, x, b) for d, x, b, h in rows if b >= 2 and x is not None]

    use_rows = vs_hand if len(vs_hand) >= 3 else all_valid
    if len(use_rows) < 2:
        return rows[0][1] if rows[0][1] else MLB_AVG_XWOBA, 0, len(vs_hand)

    dates, xwobas, bbes = zip(*use_rows[:span])
    weights = [0.7 ** i for i in range(len(xwobas))]
    total = sum(weights)
    rolling = sum(x * w for x, w in zip(xwobas, weights)) / total if total > 0 else np.mean(xwobas)

    return rolling, len(vs_hand), len(all_valid)


def predict_xwoba_matchup(conn, batter_id, pitcher_id, pitcher_stand, home_team, before_date):
    """
    Predict expected xwOBA for a batter-pitcher matchup.

    Returns: (predicted_xwoba, pitcher_xwoba, batter_xwoba, confidence)
    """
    # 1. Pitcher suppression ability (most predictive)
    p_xwoba, p_games = get_pitcher_xwoba_rolling(conn, pitcher_id, before_date)

    # 2. Batter form vs this handedness
    b_xwoba, b_vs_hand, b_total = get_batter_xwoba_rolling(conn, batter_id, before_date, pitcher_stand)

    # 3. Platoon adjustment
    # If batter has data vs this hand, use it. Otherwise, apply generic +15pt platoon
    if b_vs_hand >= 3:
        platoon_adj = 0  # already using correct split
    else:
        # Generic platoon: batters do ~15pt better vs opposite hand
        # pitcher_stand = 'L' means batter is likely righty advantage
        platoon_adj = 0.015

    # 4. Park factor
    park = PARK_FACTORS.get(home_team, 1.00)

    # 5. Blend: pitcher-centric (70%) + batter form (30%)
    # Pitcher xwOBA is more stable, so weight more
    if p_games >= 3:
        blended = (p_xwoba * 0.70 + b_xwoba * 0.30) * park + platoon_adj
    elif p_games >= 1:
        blended = (p_xwoba * 0.55 + b_xwoba * 0.45) * park + platoon_adj
    else:
        blended = b_xwoba * park + platoon_adj

    # Clamp to realistic range
    blended = max(0.150, min(0.550, blended))

    # Confidence: based on data volume
    confidence = min(1.0, (p_games * 0.3 + b_total * 0.2) / 5.0)

    return blended, p_xwoba, b_xwoba, confidence


def predict_game_xwoba(conn, game_row, before_date):
    """Predict xwOBA for all batters in a game. Returns dict with predictions."""
    import json

    h_team = game_row.get('home_team', '')
    a_team = game_row.get('away_team', '')
    hp_id = game_row.get('pp_h_starter_id', 0)
    ap_id = game_row.get('pp_a_starter_id', 0)
    hp_stand = game_row.get('pp_h_starter_hand', 'R')
    ap_stand = game_row.get('pp_a_starter_hand', 'R')

    if pd.isna(hp_id): hp_id = 0
    if pd.isna(ap_id): ap_id = 0

    # Get lineups from batter_performances for this game
    gpk = game_row.get('game_pk', 0)
    if pd.isna(gpk): gpk = 0

    h_batters = conn.execute('''
        SELECT bp.player_id, bp.batting_order, bp.player_name
        FROM batter_performances bp
        WHERE bp.game_pk = ? AND bp.team_id = (SELECT team_id FROM historico_partidos WHERE game_pk = ? LIMIT 1)
        ORDER BY bp.batting_order
    ''', (int(gpk), int(gpk))).fetchall() if gpk else []

    a_batters = conn.execute('''
        SELECT bp.player_id, bp.batting_order, bp.player_name
        FROM batter_performances bp
        WHERE bp.game_pk = ? AND bp.team_id = (SELECT CASE WHEN home_team=? THEN home_team_id ELSE away_team_id END FROM historico_partidos WHERE game_pk = ? LIMIT 1)
        ORDER BY bp.batting_order
    ''', (int(gpk), a_team, int(gpk))).fetchall() if gpk else []

    # Predict for away batters vs home pitcher
    a_predictions = []
    for bid, order, name in a_batters[:9]:
        if bid and bid > 0:
            pred, px, bx, conf = predict_xwoba_matchup(conn, int(bid), int(hp_id), hp_stand, h_team, before_date)
            a_predictions.append({
                'batter_id': bid, 'batter_name': name, 'order': order,
                'predicted_xwoba': round(pred, 3),
                'pitcher_xwoba': round(px, 3), 'batter_xwoba': round(bx, 3),
                'confidence': round(conf, 2)
            })

    # Predict for home batters vs away pitcher
    h_predictions = []
    for bid, order, name in h_batters[:9]:
        if bid and bid > 0:
            pred, px, bx, conf = predict_xwoba_matchup(conn, int(bid), int(ap_id), ap_stand, a_team, before_date)
            h_predictions.append({
                'batter_id': bid, 'batter_name': name, 'order': order,
                'predicted_xwoba': round(pred, 3),
                'pitcher_xwoba': round(px, 3), 'batter_xwoba': round(bx, 3),
                'confidence': round(conf, 2)
            })

    # Team averages
    h_avg = np.mean([p['predicted_xwoba'] for p in h_predictions]) if h_predictions else MLB_AVG_XWOBA
    a_avg = np.mean([p['predicted_xwoba'] for p in a_predictions]) if a_predictions else MLB_AVG_XWOBA

    return {
        'home_team': h_team, 'away_team': a_team,
        'home_pitcher_xwoba': round(p_xwoba, 3) if 'p_xwoba' in dir() else MLB_AVG_XWOBA,
        'away_pitcher_xwoba': round(p_xwoba, 3) if 'p_xwoba' in dir() else MLB_AVG_XWOBA,
        'home_lineup_avg_xwoba': round(h_avg, 3),
        'away_lineup_avg_xwoba': round(a_avg, 3),
        'home_batters': h_predictions,
        'away_batters': a_predictions,
    }


def predict_today(date_str=None):
    """Predict xwOBA for all games on a given date."""
    conn = sqlite3.connect(str(DB_PATH))

    if not date_str:
        from datetime import date
        date_str = date.today().isoformat()

    # Get games from schedule
    games = conn.execute('''
        SELECT game_pk, home_team, away_team, home_team_id, away_team_id
        FROM historico_partidos
        WHERE date = ? AND game_pk > 0
    ''', (date_str,)).fetchall()

    if not games:
        print(f"No games found for {date_str}")
        conn.close()
        return []

    results = []
    for gpk, h_team, a_team, h_tid, a_tid in games:
        # Get probable pitchers from pitcher_performances
        pp = conn.execute('''
            SELECT player_id, player_name, team_id
            FROM pitcher_performances
            WHERE game_pk = ? AND role = 'starter'
        ''', (gpk,)).fetchall()

        # Determine home/away by team_id match
        hp_id, hp_name, ap_id, ap_name = 0, 'TBD', 0, 'TBD'
        hp_tid, ap_tid = None, None
        for pid, pname, tid in pp:
            if tid == h_tid:
                hp_id, hp_name, hp_tid = pid, pname, tid
            elif tid == a_tid:
                ap_id, ap_name, ap_tid = pid, pname, tid

        # If team_ids are NULL in historico_partidos, try to infer from batter_performances
        if hp_tid is None or ap_tid is None:
            team_ids = conn.execute('''
                SELECT DISTINCT team_id FROM batter_performances WHERE game_pk = ?
            ''', (gpk,)).fetchall()
            if len(team_ids) >= 2:
                t1, t2 = team_ids[0][0], team_ids[1][0]
                # Assign: first team that has a starter is home (convention)
                for pid, pname, tid in pp:
                    if tid == t1 and hp_tid is None:
                        hp_id, hp_name, hp_tid = pid, pname, tid
                    elif tid == t2 and ap_tid is None:
                        ap_id, ap_name, ap_tid = pid, pname, tid
                # If still missing, assign remaining
                if hp_tid is None and ap_tid is not None:
                    for pid, pname, tid in pp:
                        if tid != ap_tid:
                            hp_id, hp_name, hp_tid = pid, pname, tid
                            break
                elif ap_tid is None and hp_tid is not None:
                    for pid, pname, tid in pp:
                        if tid != hp_tid:
                            ap_id, ap_name, ap_tid = pid, pname, tid
                            break
                h_tid, a_tid = t1, t2

        # Get pitcher hands
        hp_hand = conn.execute('''
            SELECT stand FROM savant_pitcher_daily WHERE player_id = ? ORDER BY game_date DESC LIMIT 1
        ''', (hp_id,)).fetchone() if hp_id else None
        ap_hand = conn.execute('''
            SELECT stand FROM savant_pitcher_daily WHERE player_id = ? ORDER BY game_date DESC LIMIT 1
        ''', (ap_id,)).fetchone() if ap_id else None

        hp_stand = hp_hand[0] if hp_hand else 'R'
        ap_stand = ap_hand[0] if ap_hand else 'R'

        # Get lineups from batter_performances
        h_batters = conn.execute('''
            SELECT bp.player_id, bp.batting_order, bp.player_name
            FROM batter_performances bp
            WHERE bp.game_pk = ? AND bp.team_id = ?
            ORDER BY bp.batting_order
        ''', (gpk, h_tid)).fetchall() if h_tid else []

        a_batters = conn.execute('''
            SELECT bp.player_id, bp.batting_order, bp.player_name
            FROM batter_performances bp
            WHERE bp.game_pk = ? AND bp.team_id = ?
            ORDER BY bp.batting_order
        ''', (gpk, a_tid)).fetchall() if a_tid else []

        # Predict
        game_result = {
            'game_pk': gpk, 'home_team': h_team, 'away_team': a_team,
            'home_pitcher': hp_name, 'away_pitcher': ap_name,
            'home_pitcher_id': hp_id, 'away_pitcher_id': ap_id,
            'batters': []
        }

        h_preds, a_preds = [], []

        # Away batters vs home pitcher
        for bid, order, name in a_batters[:9]:
            if bid and bid > 0 and hp_id and hp_id > 0:
                pred, px, bx, conf = predict_xwoba_matchup(conn, int(bid), int(hp_id), hp_stand, h_team, date_str)
                row = {'batter_id': bid, 'batter_name': name, 'order': order, 'team': a_team,
                       'predicted_xwoba': round(pred, 3), 'pitcher_xwoba': round(px, 3),
                       'batter_xwoba': round(bx, 3), 'confidence': round(conf, 2),
                       'vs_pitcher_hand': hp_stand}
                a_preds.append(row)

        # Home batters vs away pitcher
        for bid, order, name in h_batters[:9]:
            if bid and bid > 0 and ap_id and ap_id > 0:
                pred, px, bx, conf = predict_xwoba_matchup(conn, int(bid), int(ap_id), ap_stand, a_team, date_str)
                row = {'batter_id': bid, 'batter_name': name, 'order': order, 'team': h_team,
                       'predicted_xwoba': round(pred, 3), 'pitcher_xwoba': round(px, 3),
                       'batter_xwoba': round(bx, 3), 'confidence': round(conf, 2),
                       'vs_pitcher_hand': ap_stand}
                h_preds.append(row)

        game_result['away_lineup_xwoba'] = round(np.mean([p['predicted_xwoba'] for p in a_preds]), 3) if a_preds else 0.310
        game_result['home_lineup_xwoba'] = round(np.mean([p['predicted_xwoba'] for p in h_preds]), 3) if h_preds else 0.310
        game_result['away_batters'] = a_preds
        game_result['home_batters'] = h_preds
        game_result['matchup_advantage'] = round(game_result['home_lineup_xwoba'] - game_result['away_lineup_xwoba'], 3)

        results.append(game_result)

    conn.close()
    return results


def print_predictions(results):
    """Pretty-print xwOBA predictions."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    for game in results:
        h_team = game['home_team']
        a_team = game['away_team']
        hp_name = game['home_pitcher']
        ap_name = game['away_pitcher']

        console.print(f"\n[bold cyan]{a_team}[/] @ [bold cyan]{h_team}[/]")
        console.print(f"  Pitchers: [yellow]{ap_name}[/] (away) vs [yellow]{hp_name}[/] (home)")

        # Away lineup
        table = Table(title=f"  {a_team} lineup vs {hp_name}", show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Batter", width=20)
        table.add_column("Pred xwOBA", width=10, justify="center")
        table.add_column("Pitcher Supp", width=12, justify="center")
        table.add_column("Batter Form", width=12, justify="center")
        table.add_column("Conf", width=6, justify="center")

        for b in game['away_batters']:
            color = "green" if b['predicted_xwoba'] > 0.350 else "red" if b['predicted_xwoba'] < 0.280 else "white"
            table.add_row(
                str(b['order']), b['batter_name'][:20],
                f"[{color}]{b['predicted_xwoba']:.3f}[/]",
                f"{b['pitcher_xwoba']:.3f}",
                f"{b['batter_xwoba']:.3f}",
                f"{b['confidence']:.0%}"
            )
        console.print(table)

        # Home lineup
        table2 = Table(title=f"  {h_team} lineup vs {ap_name}", show_header=True, header_style="bold")
        table2.add_column("#", width=3)
        table2.add_column("Batter", width=20)
        table2.add_column("Pred xwOBA", width=10, justify="center")
        table2.add_column("Pitcher Supp", width=12, justify="center")
        table2.add_column("Batter Form", width=12, justify="center")
        table2.add_column("Conf", width=6, justify="center")

        for b in game['home_batters']:
            color = "green" if b['predicted_xwoba'] > 0.350 else "red" if b['predicted_xwoba'] < 0.280 else "white"
            table2.add_row(
                str(b['order']), b['batter_name'][:20],
                f"[{color}]{b['predicted_xwoba']:.3f}[/]",
                f"{b['pitcher_xwoba']:.3f}",
                f"{b['batter_xwoba']:.3f}",
                f"{b['confidence']:.0%}"
            )
        console.print(table2)

        # Summary
        adv = game['matchup_advantage']
        adv_color = "green" if adv > 0.01 else "red" if adv < -0.01 else "yellow"
        console.print(f"  Lineup avg: {a_team} {game['away_lineup_xwoba']:.3f} vs {h_team} {game['home_lineup_xwoba']:.3f} → [{adv_color}]advantage {h_team} {adv:+.3f}[/]")


def backtest_xwoba(conn, game_pk, actual_home_xwoba=None, actual_away_xwoba=None):
    """
    Backtest xwOBA prediction for a completed game.
    If actual xwoba not provided, returns predictions only.
    """
    game = conn.execute('''
        SELECT game_pk, home_team, away_team FROM historico_partidos WHERE game_pk = ?
    ''', (game_pk,)).fetchone()

    if not game:
        return None

    result = predict_game_xwoba(conn, {'game_pk': game_pk, 'home_team': game[1], 'away_team': game[2]}, game_pk)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xwOBA Prediction Module (TEST ONLY)')
    parser.add_argument('--date', type=str, default=None, help='Date to predict (YYYY-MM-DD)')
    parser.add_argument('--mode', choices=['predict', 'test'], default='predict')
    args = parser.parse_args()

    results = predict_today(args.date)
    if results:
        print_predictions(results)
    else:
        print("No games found or no data available.")
