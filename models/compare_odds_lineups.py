#!/usr/bin/env python3
"""
OMEGA MLB — Live Betting Lines vs Prop Projections Comparison Tool with Lineup Status
Queries theoddsapi, loads Prop Hunter projections, checks StatsAPI for confirmed lineups,
matches players, and groups advantages by Lineup Confirmation Status.
"""
import os
import sys
import argparse
import requests
import sqlite3
import numpy as np
from datetime import datetime
import statsapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from models.props import PropsAnalyzerV3, get_player_name

# Colors for terminal printing
C_BOLD="\033[1m"; C_CYAN="\033[96m"; C_GREEN="\033[92m"; C_YELLOW="\033[93m"
C_RED="\033[91m"; C_DIM="\033[2m"; C_RESET="\033[0m"; C_WHITE=""

def get_implied_probability(american_odds):
    """Convert American odds to implied probability (0.0 to 1.0)."""
    try:
        odds = float(american_odds)
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)
    except:
        return None

def robust_player_match(proj_name, api_name):
    """Match player name from model with player name from the odds API."""
    p_name = proj_name.lower().replace('.', '').replace(',', '').strip()
    a_name = api_name.lower().replace('.', '').replace(',', '').strip()
    
    if p_name in a_name or a_name in p_name:
        return True
        
    p_parts = p_name.split()
    a_parts = a_name.split()
    
    if len(p_parts) == 2 and len(p_parts[1]) == 1:
        last, init = p_parts[0], p_parts[1]
        if last in a_parts and any(pt.startswith(init) for pt in a_parts if pt != last):
            return True
            
    if len(a_parts) == 2 and len(a_parts[1]) == 1:
        last, init = a_parts[0], a_parts[1]
        if last in p_parts and any(pt.startswith(init) for pt in p_parts if pt != last):
            return True
            
    common_lasts = {'smith', 'johnson', 'williams', 'brown', 'jones', 'miller', 'davis', 'garcia', 'rodriguez'}
    for pt in p_parts:
        if len(pt) > 3 and pt not in common_lasts:
            if pt in a_parts:
                return True
                
    return False

def match_teams(team1, team2):
    t1 = team1.lower().replace('.', '').replace(' ', '')
    t2 = team2.lower().replace('.', '').replace(' ', '')
    if t1 in t2 or t2 in t1:
        return True
    w1 = set(team1.lower().split())
    w2 = set(team2.lower().split())
    ignore = {'los', 'angeles', 'new', 'york', 'san', 'diego', 'francisco', 'st', 'louis', 'city', 'tampa', 'bay', 'red', 'sox', 'white', 'blue', 'jays'}
    w1_clean = w1 - ignore
    w2_clean = w2 - ignore
    if w1_clean & w2_clean:
        return True
    return False

def match_game(game1, game2):
    try:
        a1, h1 = game1.split('@')
        a2, h2 = game2.split('@')
        return match_teams(a1.strip(), a2.strip()) and match_teams(h1.strip(), h2.strip())
    except:
        return False

def fetch_mlb_events(api_key):
    """Get list of upcoming MLB games and their event IDs."""
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"🚨 Error fetching MLB events: {e}")
        return []

def fetch_event_props(api_key, event_id):
    """Fetch all props for a specific game event ID."""
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
    markets = "pitcher_strikeouts,batter_hits,batter_home_runs,batter_total_bases"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': markets,
        'oddsFormat': 'american',
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {}

def main():
    parser = argparse.ArgumentParser(description='Compare Prop Hunter Projections vs Live Betting Lines (Lineup Status Aware)')
    parser.add_argument('--api-key', type=str, default=None, help='theoddsapi.com API Key')
    parser.add_argument('--date', type=str, default=None, help='Date in YYYY-MM-DD format')
    parser.add_argument('--bookmaker', type=str, default='draftkings', help='Bookmaker key')
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('THEODDSAPI_KEY') or os.environ.get('THEODDSAPI_API_KEY')
    if not api_key:
        print("🚨 ERROR: Se requiere una API Key de theoddsapi.com.")
        sys.exit(1)
        
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')
    print(f"\n{C_BOLD}{C_CYAN}OMEGA MLB — COMPARADOR DE LÍNEAS vs PROYECCIONES CON STATUS DE ALINEACIÓN ({date_str}){C_RESET}")
    print(f"Bookmaker objetivo: {args.bookmaker.upper()}\n")
    
    # 1. Check MLB starting lineups confirmation status on StatsAPI
    print("⏳ Consultando confirmación de alineaciones en MLB StatsAPI...")
    sched = statsapi.schedule(date=date_str)
    confirmed_games = set()
    partial_confirmed_games = {} # game_name -> side_confirmed
    
    for g in sched:
        pk = g['game_id']
        hn = g['home_name']
        an = g['away_name']
        game_label = f"{an} @ {hn}"
        try:
            box = statsapi.boxscore_data(pk)
            h_batters = box.get('homeBatters', [])
            a_batters = box.get('awayBatters', [])
            h_starters = [b for b in h_batters if b.get('personId', 0) > 0 and not b.get('substitution', False)]
            a_starters = [b for b in a_batters if b.get('personId', 0) > 0 and not b.get('substitution', False)]
            
            h_conf = len(h_starters) >= 8
            a_conf = len(a_starters) >= 8
            
            if h_conf and a_conf:
                confirmed_games.add(game_label)
            elif h_conf:
                partial_confirmed_games[game_label] = 'HOME'
            elif a_conf:
                partial_confirmed_games[game_label] = 'AWAY'
        except Exception as e:
            pass
            
    print(f"✅ Alineaciones confirmadas para {len(confirmed_games)} partidos, parcialmente para {len(partial_confirmed_games)} partidos.")
    
    # 2. Load Prop Projections
    print("⏳ Ejecutando Prop Hunter local para obtener proyecciones...")
    analyzer = PropsAnalyzerV3()
    proj_results = analyzer.scan(date_str)
    conn = analyzer.conn
    
    projections = {}
    for cat, players in proj_results.items():
        for p in players:
            if cat == 'K (Pitcher)':
                projections[(cat, p['pitcher'])] = p['proj_k']
            else:
                pname = get_player_name(conn, p['player_id'])
                projections[(cat, pname)] = p['prob']
                
    print(f"✅ Proyecciones cargadas: {len(projections)} métricas.")
    
    # 3. Fetch MLB Events and their Props
    print("⏳ Obteniendo partidos de MLB programados...")
    events = fetch_mlb_events(api_key)
    if not events:
        print("❌ No se encontraron partidos de MLB.")
        sys.exit(1)
        
    print(f"✅ Se encontraron {len(events)} partidos. Obteniendo líneas de props juego por juego...")
    
    markets_mapping = {
        'pitcher_strikeouts': 'K (Pitcher)',
        'batter_hits': 'HITS (1+)',
        'batter_home_runs': 'HR (1+)',
        'batter_total_bases': 'TB (1.5+)'
    }
    
    edges = []
    
    for i, event in enumerate(events, 1):
        event_id = event['id']
        teams = f"{event['away_team']} @ {event['home_team']}"
        print(f"   [{i}/{len(events)}] Buscando props para: {teams}...", end="\r")
        
        event_data = fetch_event_props(api_key, event_id)
        if not event_data:
            continue
            
        bookmakers = event_data.get('bookmakers', [])
        target_bm = None
        for bm in bookmakers:
            if bm['key'] == args.bookmaker:
                target_bm = bm
                break
        if not target_bm and bookmakers:
            target_bm = bookmakers[0]
            
        if not target_bm:
            continue
            
        # Determine lineup status for this event
        lineup_status = 'ESTIMADA'
        for cg in confirmed_games:
            if match_game(teams, cg):
                lineup_status = 'CONFIRMADA'
                break
        if lineup_status == 'ESTIMADA':
            for pcg, side in partial_confirmed_games.items():
                if match_game(teams, pcg):
                    lineup_status = f'PARCIAL ({side})'
                    break
            
        for market in target_bm.get('markets', []):
            m_key = market['key']
            if m_key not in markets_mapping:
                continue
            cat = markets_mapping[m_key]
            
            for outcome in market.get('outcomes', []):
                name_val = outcome.get('name', '')
                desc_val = outcome.get('description', '')
                
                if name_val in {'Over', 'Under', 'Yes', 'No'}:
                    player_name = desc_val
                    bet_type = name_val
                elif desc_val in {'Over', 'Under', 'Yes', 'No'}:
                    player_name = name_val
                    bet_type = desc_val
                else:
                    if any(x in name_val for x in ['Over', 'Under', 'Yes', 'No']):
                        player_name = desc_val
                        bet_type = name_val
                    else:
                        player_name = name_val
                        bet_type = desc_val
                
                odds = outcome.get('price')
                line = outcome.get('point')
                
                if not player_name:
                    continue
                    
                matched_proj_key = None
                for (p_cat, p_name) in projections:
                    if p_cat == cat and robust_player_match(p_name, player_name):
                        matched_proj_key = (p_cat, p_name)
                        break
                        
                if not matched_proj_key:
                    continue
                    
                p_val = projections[matched_proj_key]
                p_name = matched_proj_key[1]
                
                if cat == 'K (Pitcher)':
                    if line is not None:
                        diff = p_val - line
                        if (bet_type.lower() == 'over' and diff > 0) or (bet_type.lower() == 'under' and diff < 0):
                            edges.append({
                                'Player': p_name,
                                'Category': 'Ponches (K)',
                                'Type': bet_type.upper(),
                                'Line': f"{line}",
                                'Odds': f"{odds:+d}",
                                'Proj': f"{p_val:.1f} Ks",
                                'Edge': abs(diff),
                                'EdgeStr': f"{abs(diff):+.1f} Ks",
                                'Game': teams,
                                'Status': lineup_status
                            })
                else:
                    implied_prob = get_implied_probability(odds)
                    if implied_prob is not None:
                        if 'yes' in bet_type.lower() or 'over' in bet_type.lower():
                            edge = p_val - implied_prob
                            if edge > 0:
                                edges.append({
                                    'Player': p_name,
                                    'Category': cat.replace(' (1+)', '').replace(' (1.5+)', ''),
                                    'Type': 'SÍ / OVER',
                                    'Line': '0.5' if 'hits' in m_key or 'home_runs' in m_key else '1.5',
                                    'Odds': f"{odds:+d}",
                                    'Proj': f"{p_val:.1%}",
                                    'Edge': edge,
                                    'EdgeStr': f"{edge:+.1%}",
                                    'Game': teams,
                                    'Status': lineup_status
                                })
                                
    print(f"\n✅ Análisis de props completado.")
    conn.close()
    
    if not edges:
        print("\n❌ No se encontraron ventajas (edges positivos) para las proyecciones y líneas actuales.")
        return
        
    edges.sort(key=lambda x: x['Edge'], reverse=True)
    
    # Deduplicate edges (since double headers/duplicates can happen in the API)
    seen = set()
    deduped_edges = []
    for e in edges:
        key = (e['Player'], e['Category'], e['Type'], e['Line'], e['Game'])
        if key not in seen:
            seen.add(key)
            deduped_edges.append(e)
            
    # Group into Confirmed and Estimated
    confirmed = [e for e in deduped_edges if 'CONFIRMADA' in e['Status']]
    partial = [e for e in deduped_edges if 'PARCIAL' in e['Status']]
    estimated = [e for e in deduped_edges if 'ESTIMADA' in e['Status']]
    
    def print_table(title, items):
        print(f"\n{C_BOLD}{C_GREEN}=== {title} ==={C_RESET}")
        print(f"{'JUGADOR':<22} {'PROP':<12} {'TIPO':<10} {'LÍNEA':<6} {'MOMIO':<6} {'PROYECCIÓN':<11} {'VENTAJA (EDGE)':<15} {'PARTIDO'}")
        print("-" * 115)
        for e in items:
            color = C_GREEN if e['Edge'] >= 0.08 or (e['Category'] == 'Ponches (K)' and e['Edge'] >= 1.0) else C_WHITE
            print(f"{e['Player'][:21]:<22} {e['Category']:<12} {e['Type']:<10} {e['Line']:<6} {e['Odds']:<6} {e['Proj']:<11} {color}{e['EdgeStr']:<15}{C_RESET} {e['Game']}")
        print("-" * 115)
        
    if confirmed:
        print_table("VENTAJAS CON ALINEACIÓN CONFIRMADA (Seguridad Roster)", confirmed)
    if partial:
        print_table("VENTAJAS CON ALINEACIÓN PARCIAL (Un equipo confirmado)", partial)
    if estimated:
        print_table("VENTAJAS CON ALINEACIÓN ESTIMADA (Proyecciones de Roster)", estimated)
        
    print(f"\n{C_DIM}Métricas basadas en proyecciones XGBoost y cuotas de {args.bookmaker.upper()}.{C_RESET}\n")

if __name__ == '__main__':
    main()
