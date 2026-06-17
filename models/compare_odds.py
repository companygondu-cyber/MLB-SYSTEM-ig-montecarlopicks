#!/usr/bin/env python3
"""
OMEGA MLB — Live Betting Lines vs Prop Projections Comparison Tool
Queries theoddsapi (handling event-level queries for props), loads Prop Hunter projections, 
matches players, and identifies the largest mathematical advantages (edges).
"""
import os
import sys
import argparse
import requests
import sqlite3
import numpy as np
from datetime import datetime

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
    """
    Match player name from model (e.g. 'Harper, B' or 'Witt Jr.')
    with player name from the odds API (e.g. 'Bryce Harper' or 'Bobby Witt Jr.').
    """
    p_name = proj_name.lower().replace('.', '').replace(',', '').strip()
    a_name = api_name.lower().replace('.', '').replace(',', '').strip()
    
    # Simple direct match or substring
    if p_name in a_name or a_name in p_name:
        return True
        
    # Split names into parts
    p_parts = p_name.split()
    a_parts = a_name.split()
    
    # If one is last_name, first_initial (like 'harper b')
    if len(p_parts) == 2 and len(p_parts[1]) == 1:
        last, init = p_parts[0], p_parts[1]
        if last in a_parts and any(pt.startswith(init) for pt in a_parts if pt != last):
            return True
            
    if len(a_parts) == 2 and len(a_parts[1]) == 1:
        last, init = a_parts[0], a_parts[1]
        if last in p_parts and any(pt.startswith(init) for pt in p_parts if pt != last):
            return True
            
    # Try last name matching for distinct last names
    common_lasts = {'smith', 'johnson', 'williams', 'brown', 'jones', 'miller', 'davis', 'garcia', 'rodriguez'}
    for pt in p_parts:
        if len(pt) > 3 and pt not in common_lasts:
            if pt in a_parts:
                return True
                
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
    parser = argparse.ArgumentParser(description='Compare Prop Hunter Projections vs Live Betting Lines')
    parser.add_argument('--api-key', type=str, default=None, help='theoddsapi.com API Key')
    parser.add_argument('--date', type=str, default=None, help='Date in YYYY-MM-DD format')
    parser.add_argument('--bookmaker', type=str, default='draftkings', help='Bookmaker key (draftkings, fanduel, betmgm, etc.)')
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('THEODDSAPI_KEY') or os.environ.get('THEODDSAPI_API_KEY')
    if not api_key:
        print("🚨 ERROR: Se requiere una API Key de theoddsapi.com.")
        print("Puedes proporcionarla con el argumento --api-key o configurando la variable de entorno THEODDSAPI_KEY.")
        sys.exit(1)
        
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')
    print(f"\n{C_BOLD}{C_CYAN}OMEGA MLB — COMPARADOR DE LÍNEAS vs PROYECCIONES ({date_str}){C_RESET}")
    print(f"Bookmaker objetivo: {args.bookmaker.upper()}\n")
    
    # 1. Load Prop Projections
    print("⏳ Ejecutando Prop Hunter local para obtener proyecciones...")
    analyzer = PropsAnalyzerV3()
    proj_results = analyzer.scan(date_str)
    conn = analyzer.conn
    
    # We will map: (Category, Player Name) -> Projection
    projections = {}
    for cat, players in proj_results.items():
        for p in players:
            if cat == 'K (Pitcher)':
                projections[(cat, p['pitcher'])] = p['proj_k']
            else:
                pname = get_player_name(conn, p['player_id'])
                projections[(cat, pname)] = p['prob']
                
    print(f"✅ Proyecciones cargadas: {len(projections)} métricas.")
    
    # 2. Fetch MLB Events and their Props
    print("⏳ Obteniendo partidos de MLB programados...")
    events = fetch_mlb_events(api_key)
    if not events:
        print("❌ No se encontraron partidos de MLB.")
        sys.exit(1)
        
    print(f"✅ Se encontraron {len(events)} partidos. Obteniendo líneas de props juego por juego...")
    
    # Maps market keys in theoddsapi to our category names
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
            
        for market in target_bm.get('markets', []):
            m_key = market['key']
            if m_key not in markets_mapping:
                continue
            cat = markets_mapping[m_key]
            
            for outcome in market.get('outcomes', []):
                name_val = outcome.get('name', '')
                desc_val = outcome.get('description', '')
                
                # Robust extraction of player name and bet type
                if name_val in {'Over', 'Under', 'Yes', 'No'}:
                    player_name = desc_val
                    bet_type = name_val
                elif desc_val in {'Over', 'Under', 'Yes', 'No'}:
                    player_name = name_val
                    bet_type = desc_val
                else:
                    # Check if Over/Under is contained
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
                    
                # Find matching projection
                matched_proj_key = None
                for (p_cat, p_name) in projections:
                    if p_cat == cat and robust_player_match(p_name, player_name):
                        matched_proj_key = (p_cat, p_name)
                        break
                        
                if not matched_proj_key:
                    continue
                    
                p_val = projections[matched_proj_key]
                p_name = matched_proj_key[1]
                
                # Compute advantage
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
                                'Game': teams
                            })
                else:
                    # Yes/No props (Hits, HR, TB)
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
                                    'Game': teams
                                })
                                
    print(f"\n✅ Análisis de props completado.")
    conn.close()
    
    # 4. Print results sorted by Edge
    if not edges:
        print("\n❌ No se encontraron ventajas (edges positivos) para las proyecciones y líneas actuales.")
        return
        
    edges.sort(key=lambda x: x['Edge'], reverse=True)
    
    print(f"\n{C_BOLD}{C_GREEN}=== VENTAJAS DETECTADAS (Ordenadas de Mayor a Menor) ==={C_RESET}")
    print(f"{'JUGADOR':<22} {'PROP':<12} {'TIPO':<10} {'LÍNEA':<6} {'MOMIO':<6} {'PROYECCIÓN':<11} {'VENTAJA (EDGE)':<15} {'PARTIDO'}")
    print("-" * 110)
    for e in edges:
        color = C_GREEN if e['Edge'] >= 0.08 or (e['Category'] == 'Ponches (K)' and e['Edge'] >= 1.0) else C_WHITE
        print(f"{e['Player'][:21]:<22} {e['Category']:<12} {e['Type']:<10} {e['Line']:<6} {e['Odds']:<6} {e['Proj']:<11} {color}{e['EdgeStr']:<15}{C_RESET} {e['Game']}")
    print("-" * 110)
    print(f"\n{C_DIM}Métricas basadas en proyecciones XGBoost y cuotas de {args.bookmaker.upper()}.{C_RESET}\n")

if __name__ == '__main__':
    main()
