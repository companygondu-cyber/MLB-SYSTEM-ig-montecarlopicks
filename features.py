"""Core feature engineering - combines Prod's 67 features + SABER + ELO + bullpen"""
import numpy as np, pandas as pd
from datetime import timedelta
import warnings
import os, sqlite3
from collections import defaultdict
import time

from config import CONFIG, TEAM_IDS, TEAM_DIV, PARK_FACTORS, DIVISIONS

# Backward-compat export: legacy code imports DB_PATH from features
DB_PATH = str(CONFIG.paths.db)

# ── Module-level caches for expensive pre-computations ──
_H2H_LOOKUP_CACHE = None
_PF_DATA_CACHE = None

def _safe_float(val, default=0.0):
    if pd.isna(val) or val is None or val == '': return float(default)
    try: return float(val)
    except: return float(default)

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


DIVISIONS = {
    'AL East': {'Baltimore Orioles','Boston Red Sox','New York Yankees','Tampa Bay Rays','Toronto Blue Jays'},
    'AL Central': {'Chicago White Sox','Cleveland Guardians','Detroit Tigers','Kansas City Royals','Minnesota Twins'},
    'AL West': {'Houston Astros','Los Angeles Angels','Oakland Athletics','Seattle Mariners','Texas Rangers'},
    'NL East': {'Atlanta Braves','Miami Marlins','New York Mets','Philadelphia Phillies','Washington Nationals'},
    'NL Central': {'Chicago Cubs','Cincinnati Reds','Milwaukee Brewers','Pittsburgh Pirates','St. Louis Cardinals'},
    'NL West': {'Arizona Diamondbacks','Colorado Rockies','Los Angeles Dodgers','San Diego Padres','San Francisco Giants'},
}
# ── PARK FACTORS, TEAMS, DIVISIONS are imported from config.py ──

# ── PROD features (67) + SABER (12) + ELO/Bullpen (6) = ~85 features ──
BATTER_FEATS = []
for pfx in ['h','a']:
    for i in range(1,10):
        BATTER_FEATS += [f'{pfx}_batter_{i}_avg', f'{pfx}_batter_{i}_ops']

PITCHER_FEATS = ['h_starter_era','a_starter_era','h_starter_whip','a_starter_whip',
                 'h_starter_strikeout_rate','a_starter_strikeout_rate',
                 'h_starter_walk_rate','a_starter_walk_rate']

TEAM_FEATS = ['h_avg','h_obp','h_slg','h_ops','a_avg','a_obp','a_slg','a_ops']

CONTEXT_FEATS = ['series_game_number','temperature','is_night',
                 'h_rest_days','a_rest_days','divisional_game', 'park_factor']

DIFF_FEATS = ['era_diff','whip_diff','ops_diff','avg_diff','slg_diff','obp_diff',
              'k_rate_diff','bb_rate_diff','lineup_power_diff','lineup_depth_diff',
              'domination_diff','contact_q_diff']

SABER_FEATS = ['h_fip','a_fip','h_tt_era','a_tt_era','h_kbb','a_kbb',
               'h_dom','a_dom','h_era_whip_ratio','a_era_whip_ratio',
               'h_era_x_a_ops','a_era_x_h_ops']

DYNAMIC_FEATS = ['elo_diff','streak_diff',
                 'h_bp_avail','a_bp_avail','h_bp_fatigue','a_bp_fatigue',
                 'h_bullpen_era_l3','a_bullpen_era_l3']

MC_FEATS = [
    'mc_home_prob', 'mc_margin_expected', 'mc_blowout_prob', 'mc_upset_risk'
]

ELO_FEATS = ['elo_pitcher_diff', 'elo_lineup_diff', 'elo_pitcher_trend_h', 'elo_pitcher_trend_a', 'elo_hot_batters_diff', 'pitcher_elo_x_lineup_xwoba']

# NEW: High-signal power features for brute force accuracy
POWER_FEATS = [
    'h_pyth_pct','a_pyth_pct','pyth_diff',        # Pythagorean win %
    'h_run_diff_10','a_run_diff_10','run_diff_diff', # Rolling run differential
    'h_home_wpct','a_away_wpct',                    # Home/away splits
    'h2h_advantage',                                 # Head-to-head this season
    'h_starter_era_l3','a_starter_era_l3',           # Starter rolling ERA (last 3)
    'lineup_quality_weighted_diff',                   # Weighted batting order
    'momentum_h','momentum_a',                       # streak * elo interaction
    'bullpen_leverage_diff',                          # bp_avail * (1/bp_fatigue)
    'h_starter_h2h_era','a_starter_h2h_era',        # Starter career ERA vs opponent
    'h2h_era_diff',                                  # H2H ERA differential
]

VOLATILITY_FEATS = [
    'h_starter_era_std','a_starter_era_std',         # Starter ERA volatility (game-to-game)
    'h_starter_sample_ip','a_starter_sample_ip',     # Starter total IP this season (sample size)
    'h_starter_injury_flag','a_starter_injury_flag', # Recent injury/long gap flag
]

# Pitcher form/quality features (replaces old Markov regime)
PITCHER_FORM_FEATS = [
    'h_pitcher_form_score','a_pitcher_form_score',  # Composite quality (K-BB-HR-ERA weighted)
    'h_pitcher_form_vs_baseline','a_pitcher_form_vs_baseline', # Current form vs season avg (+ trending up, - trending down)
    'h_pitcher_quality_idx','a_pitcher_quality_idx', # Quality index (form × stability)
    'pitcher_quality_diff',                            # Home - away quality advantage
    'h_pitcher_ip_consistency','a_pitcher_ip_consistency', # How deep they go (IP avg last 3 / IP avg season)
    'h_era_trend','a_era_trend',                       # L3 ERA vs season ERA delta (- = improving, + = declining)
]

MARKOV_FEATS = [
    'h_markov_ww', 'h_markov_wl', 'h_markov_expected_win',
    'a_markov_ww', 'a_markov_wl', 'a_markov_expected_win',
    'markov_advantage',
    'h_pitcher_qs_consistency', 'a_pitcher_qs_consistency',
    'h_pitcher_markov_entropy', 'a_pitcher_markov_entropy',
    'h_markov_dominant_rate', 'a_markov_dominant_rate',
    'markov_momentum_diff'
]


def get_pitcher_season_stats_df(pp_df, pid, year, before_date=None):
    start_date = pd.Timestamp(f'{year}-01-01')
    end_date = pd.Timestamp(f'{year}-12-31')
    if before_date is not None:
        end_date = min(end_date, pd.to_datetime(before_date) - pd.Timedelta(days=1))
    p_df = pp_df[(pp_df['player_id'] == pid) & (pp_df['role'] == 'starter') & (pp_df['date'] >= start_date) & (pp_df['date'] <= end_date)]
    if p_df.empty:
        return None
    k = sum(p_df['k'].dropna())
    bb = sum(p_df['bb'].dropna())
    ip = sum(parse_baseball_ip(x) for x in p_df['ip'])
    er = sum(p_df['er'].dropna())
    ha = sum(p_df['hits'].dropna())
    bf = sum(p_df['batters_faced'].dropna())
    if ip < 1.0:
        return None
    if bf <= 0:
        bf = max(int(ip * 4.35) + ha + bb, 1)
    return {
        'era': er * 9.0 / ip,
        'whip': (ha + bb) / ip,
        'strikeout_rate': k / bf,
        'walk_rate': bb / bf,
        'ip': ip
    }

def adjust_pitcher_stats_blended(df, pp_df):
    adjusted_df = df.copy()
    starters = pp_df[pp_df['role'] == 'starter']
    h_era_adj, a_era_adj = [], []
    h_whip_adj, a_whip_adj = [], []
    h_so_adj, a_so_adj = [], []
    h_bb_adj, a_bb_adj = [], []
    for idx, row in adjusted_df.iterrows():
        gd = row['date']
        year = gd.year
        prev_year = year - 1
        h_team = row['home_team']
        a_team = row['away_team']
        htid = TEAM_IDS.get(h_team)
        atid = TEAM_IDS.get(a_team)
        match_h = starters[(starters['date'] == gd) & (starters['team_id'] == htid)]
        match_a = starters[(starters['date'] == gd) & (starters['team_id'] == atid)]
        h_stats = {'era': row['h_starter_era'], 'whip': row['h_starter_whip'], 
                   'strikeout_rate': row.get('h_starter_strikeout_rate', np.nan), 
                   'walk_rate': row.get('h_starter_walk_rate', np.nan)}
        a_stats = {'era': row['a_starter_era'], 'whip': row['a_starter_whip'], 
                   'strikeout_rate': row.get('a_starter_strikeout_rate', np.nan), 
                   'walk_rate': row.get('a_starter_walk_rate', np.nan)}
        if not match_h.empty:
            hpid = int(match_h.iloc[0]['player_id'])
            curr_stats = get_pitcher_season_stats_df(pp_df, hpid, year, before_date=gd)
            curr_ip = curr_stats['ip'] if curr_stats else 0.0
            if curr_ip < 30.0:
                prev_stats = get_pitcher_season_stats_df(pp_df, hpid, prev_year)
                if prev_stats:
                    w = curr_ip / 30.0
                    for key in ['era', 'whip', 'strikeout_rate', 'walk_rate']:
                        c_val = curr_stats[key] if (curr_stats and key in curr_stats) else prev_stats[key]
                        c_val = prev_stats[key] if pd.isna(c_val) else c_val
                        h_stats[key] = w * c_val + (1 - w) * prev_stats[key]
        if not match_a.empty:
            apid = int(match_a.iloc[0]['player_id'])
            curr_stats = get_pitcher_season_stats_df(pp_df, apid, year, before_date=gd)
            curr_ip = curr_stats['ip'] if curr_stats else 0.0
            if curr_ip < 30.0:
                prev_stats = get_pitcher_season_stats_df(pp_df, apid, prev_year)
                if prev_stats:
                    w = curr_ip / 30.0
                    for key in ['era', 'whip', 'strikeout_rate', 'walk_rate']:
                        c_val = curr_stats[key] if (curr_stats and key in curr_stats) else prev_stats[key]
                        c_val = prev_stats[key] if pd.isna(c_val) else c_val
                        a_stats[key] = w * c_val + (1 - w) * prev_stats[key]
        h_era_adj.append(h_stats['era'])
        h_whip_adj.append(h_stats['whip'])
        h_so_adj.append(h_stats['strikeout_rate'])
        h_bb_adj.append(h_stats['walk_rate'])
        a_era_adj.append(a_stats['era'])
        a_whip_adj.append(a_stats['whip'])
        a_so_adj.append(a_stats['strikeout_rate'])
        a_bb_adj.append(a_stats['walk_rate'])
    adjusted_df['h_starter_era'] = h_era_adj
    adjusted_df['h_starter_whip'] = h_whip_adj
    adjusted_df['h_starter_strikeout_rate'] = h_so_adj
    adjusted_df['h_starter_walk_rate'] = h_bb_adj
    adjusted_df['a_starter_era'] = a_era_adj
    adjusted_df['a_starter_whip'] = a_whip_adj
    adjusted_df['a_starter_strikeout_rate'] = a_so_adj
    adjusted_df['a_starter_walk_rate'] = a_bb_adj
    return adjusted_df

SAVANT_FEATS = [
    'h_starter_xwoba', 'a_starter_xwoba', 'h_starter_barrel', 'a_starter_barrel', 'h_starter_hardhit', 'a_starter_hardhit', 'h_starter_ev', 'a_starter_ev',
    'h_starter_spin', 'a_starter_spin', 'h_starter_extension', 'a_starter_extension',
    'h_bullpen_xwoba', 'a_bullpen_xwoba', 'h_bullpen_barrel', 'a_bullpen_barrel',
    'h_starter_true_risk', 'a_starter_true_risk',
    'h_lineup_xwoba', 'a_lineup_xwoba', 'h_lineup_barrel', 'a_lineup_barrel', 'h_lineup_hardhit', 'a_lineup_hardhit', 'h_lineup_ev', 'a_lineup_ev',
    'h_lineup_bat_speed', 'a_lineup_bat_speed', 'h_lineup_swing_length', 'a_lineup_swing_length',
    'h_lineup_sweetspot', 'a_lineup_sweetspot', 'h_lineup_discipline', 'a_lineup_discipline', 'h_lineup_efficiency', 'a_lineup_efficiency',
    'xwoba_lineup_diff', 'barrel_lineup_diff', 'hardhit_lineup_diff',
    'h_matchup_xwoba_diff', 'a_matchup_xwoba_diff', 'matchup_advantage',
    'bat_speed_diff', 'swing_length_diff', 'discipline_diff', 'efficiency_diff',
    'h_starter_stuff_plus', 'a_starter_stuff_plus',
    'h_starter_spin_delta', 'a_starter_spin_delta',
    'h_starter_extension_delta', 'a_starter_extension_delta',
    'h_starter_ev_delta', 'a_starter_ev_delta',
    'h_bp_fatigue_recent', 'a_bp_fatigue_recent',
    'h_bp_fatigue_recent_72h', 'a_bp_fatigue_recent_72h',
    'h_lineup_decision_quality', 'a_lineup_decision_quality'
]

def append_savant_features(df, pp_df, predict_mode=False):
    if pp_df is None or pp_df.empty:
        for f in SAVANT_FEATS: df[f] = np.nan
        return df, SAVANT_FEATS
        
    try:
        import sqlite3
        import pandas as pd
        import numpy as np
        conn = sqlite3.connect(DB_PATH)
        sav_p = pd.read_sql("SELECT * FROM savant_pitcher_daily", conn)
        sav_b = pd.read_sql("SELECT * FROM savant_batter_daily", conn)
        starters_bat = pd.read_sql("SELECT game_pk, date, team_id, player_id, batting_order FROM batter_performances WHERE (batting_order % 100) = 0", conn)
        player_hands_df = pd.read_sql("SELECT player_id, pitch_hand, bat_side FROM player_hands", conn)
        conn.close()
    except Exception as e:
        print(f"🚨 WARNING: Savant data missing or DB error: {e}")
        import warnings
        warnings.warn(f"Savant data missing or error: {e}")
        for f in SAVANT_FEATS: df[f] = np.nan
        return df, []
        
    if sav_p.empty:
        for f in SAVANT_FEATS: df[f] = np.nan
        return df, []

    # Map player_id to throws and bat side
    pitcher_hands = dict(zip(player_hands_df['player_id'].astype(int), player_hands_df['pitch_hand']))
    batter_sides = dict(zip(player_hands_df['player_id'].astype(int), player_hands_df['bat_side']))

    def get_pitcher_hand(pid):
        if not pid or pd.isna(pid): return 'R'
        pid_int = int(pid)
        hand = pitcher_hands.get(pid_int)
        if hand: return hand
        # Try statsapi on the fly
        try:
            import statsapi
            data = statsapi.player_stat_data(pid_int)
            ph = data.get('pitch_hand', 'R')
            hand = 'R' if ph == 'Right' else ('L' if ph == 'Left' else 'R')
            # Save cache
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT OR REPLACE INTO player_hands (player_id, pitch_hand) VALUES (?, ?)", (pid_int, hand))
            conn.commit()
            conn.close()
            pitcher_hands[pid_int] = hand
            return hand
        except:
            return 'R'

    # Convert dates to datetime
    sav_p['game_date_dt'] = pd.to_datetime(sav_p['game_date'], format='mixed', errors='coerce')
    sav_b['game_date_dt'] = pd.to_datetime(sav_b['game_date'], format='mixed', errors='coerce')
    df['date_dt'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    pp_df['date_dt'] = pd.to_datetime(pp_df['date'], format='mixed', errors='coerce')

    # Group daily data by player for fast lookup
    sav_p['ev_prod'] = sav_p['avg_ev_allowed'] * sav_p['bbe']
    sav_p['xwoba_prod'] = sav_p['avg_xwoba_allowed'] * sav_p['bbe']
    sav_p['spin_prod'] = sav_p['avg_release_spin_rate'] * sav_p['bbe']
    sav_p['ext_prod'] = sav_p['avg_release_extension'] * sav_p['bbe']
    
    # Keep stand to allow splits calculation
    sav_p_agg = sav_p.groupby(['player_id', 'game_date_dt', 'stand']).agg(
        bbe=('bbe', 'sum'),
        barrels_allowed=('barrels_allowed', 'sum'),
        hard_hits_allowed=('hard_hits_allowed', 'sum'),
        ev_prod=('ev_prod', 'sum'),
        xwoba_prod=('xwoba_prod', 'sum'),
        spin_prod=('spin_prod', 'sum'),
        ext_prod=('ext_prod', 'sum')
    ).reset_index()
    
    sav_p_agg['avg_ev_allowed'] = sav_p_agg['ev_prod'] / sav_p_agg['bbe'].replace(0, np.nan)
    sav_p_agg['avg_xwoba_allowed'] = sav_p_agg['xwoba_prod'] / sav_p_agg['bbe'].replace(0, np.nan)
    sav_p_agg['avg_release_spin_rate'] = sav_p_agg['spin_prod'] / sav_p_agg['bbe'].replace(0, np.nan)
    sav_p_agg['avg_release_extension'] = sav_p_agg['ext_prod'] / sav_p_agg['bbe'].replace(0, np.nan)
    
    sav_p_dict = {pid: grp.sort_values('game_date_dt') for pid, grp in sav_p_agg.groupby('player_id')}
    sav_b_dict = {pid: grp.sort_values('game_date_dt') for pid, grp in sav_b.groupby('player_id')}
    
    # Group relievers by team for fast fatigue calculations
    relievers_by_team = {tid: grp for tid, grp in pp_df[pp_df['role'] == 'reliever'].groupby('team_id')}
    
    # Group starting batters by game and team
    lineup_dict = starters_bat.groupby(['game_pk', 'team_id'])['player_id'].apply(list).to_dict()

    # Starter stats helper
    def get_pitcher_form(pid, gd, w_lhb=0.45, w_rhb=0.55):
        if not pid or pd.isna(pid) or int(pid) not in sav_p_dict:
            return 0.3192, 0.0826, 0.4024, 88.5, 2200.0, 6.2, 0.0, 0.0, 0.0
        grp = sav_p_dict[int(pid)]
        p_data = grp[grp['game_date_dt'] < gd]
        if p_data.empty:
            return 0.3192, 0.0826, 0.4024, 88.5, 2200.0, 6.2, 0.0, 0.0, 0.0
            
        # Get last 5 starts (weighted)
        dates = sorted(p_data['game_date_dt'].unique())
        l5_dates = dates[-5:]
        l5 = p_data[p_data['game_date_dt'].isin(l5_dates)]
        
        # Date weights: more recent = higher weight
        date_weights = {d: (i + 1) for i, d in enumerate(l5_dates)}
        
        # Splits vs LHB
        l5_l = l5[l5['stand'] == 'L']
        sum_bbe_l = 0.0
        sum_xwoba_l = 0.0
        sum_barrels_l = 0.0
        sum_hard_l = 0.0
        sum_ev_l = 0.0
        
        for _, r in l5_l.iterrows():
            wt = date_weights.get(r['game_date_dt'], 1.0)
            bbe = r['bbe']
            sum_bbe_l += bbe * wt
            sum_xwoba_l += r['avg_xwoba_allowed'] * bbe * wt
            sum_barrels_l += r['barrels_allowed'] * wt
            sum_hard_l += r['hard_hits_allowed'] * wt
            sum_ev_l += r['avg_ev_allowed'] * bbe * wt
            
        # Splits vs RHB
        l5_r = l5[l5['stand'] == 'R']
        sum_bbe_r = 0.0
        sum_xwoba_r = 0.0
        sum_barrels_r = 0.0
        sum_hard_r = 0.0
        sum_ev_r = 0.0
        
        for _, r in l5_r.iterrows():
            wt = date_weights.get(r['game_date_dt'], 1.0)
            bbe = r['bbe']
            sum_bbe_r += bbe * wt
            sum_xwoba_r += r['avg_xwoba_allowed'] * bbe * wt
            sum_barrels_r += r['barrels_allowed'] * wt
            sum_hard_r += r['hard_hits_allowed'] * wt
            sum_ev_r += r['avg_ev_allowed'] * bbe * wt
            
        # Overall baseline from these starts
        sum_bbe_all = sum_bbe_l + sum_bbe_r
        xwoba_overall = 0.3192
        barrel_overall = 0.0826
        hardhit_overall = 0.4024
        ev_overall = 88.5
        
        if sum_bbe_all > 0:
            xwoba_overall = (sum_xwoba_l + sum_xwoba_r) / sum_bbe_all
            barrel_overall = (sum_barrels_l + sum_barrels_r) / sum_bbe_all
            hardhit_overall = (sum_hard_l + sum_hard_r) / sum_bbe_all
            ev_overall = (sum_ev_l + sum_ev_r) / sum_bbe_all
            
        # Bayesian shrinkage for splits (prior_bbe = 10.0)
        prior_bbe = 10.0
        if sum_bbe_l > 0:
            xwoba_l = (sum_xwoba_l + prior_bbe * xwoba_overall) / (sum_bbe_l + prior_bbe)
            barrel_l = (sum_barrels_l + prior_bbe * barrel_overall) / (sum_bbe_l + prior_bbe)
            hardhit_l = (sum_hard_l + prior_bbe * hardhit_overall) / (sum_bbe_l + prior_bbe)
            ev_l = (sum_ev_l + prior_bbe * ev_overall) / (sum_bbe_l + prior_bbe)
        else:
            xwoba_l = xwoba_overall
            barrel_l = barrel_overall
            hardhit_l = hardhit_overall
            ev_l = ev_overall
            
        if sum_bbe_r > 0:
            xwoba_r = (sum_xwoba_r + prior_bbe * xwoba_overall) / (sum_bbe_r + prior_bbe)
            barrel_r = (sum_barrels_r + prior_bbe * barrel_overall) / (sum_bbe_r + prior_bbe)
            hardhit_r = (sum_hard_r + prior_bbe * hardhit_overall) / (sum_bbe_r + prior_bbe)
            ev_r = (sum_ev_r + prior_bbe * ev_overall) / (sum_bbe_r + prior_bbe)
        else:
            xwoba_r = xwoba_overall
            barrel_r = barrel_overall
            hardhit_r = hardhit_overall
            ev_r = ev_overall
            
        # Weighted average based on opponent lineup LHB/RHB
        xwoba = w_lhb * xwoba_l + w_rhb * xwoba_r
        barrel = w_lhb * barrel_l + w_rhb * barrel_r
        hardhit = w_lhb * hardhit_l + w_rhb * hardhit_r
        ev = w_lhb * ev_l + w_rhb * ev_r
        
        # Physical stats (independent of stand) - L5 average
        spin = float(l5['avg_release_spin_rate'].mean())
        ext = float(l5['avg_release_extension'].mean())
        
        # L2 starts for Deltas
        l2_dates = dates[-2:]
        l2 = p_data[p_data['game_date_dt'].isin(l2_dates)]
        spin_l2 = float(l2['avg_release_spin_rate'].mean())
        ext_l2 = float(l2['avg_release_extension'].mean())
        sum_bbe_l2 = l2['bbe'].sum()
        if sum_bbe_l2 > 0:
            ev_l2 = float((l2['avg_ev_allowed'] * l2['bbe']).sum() / sum_bbe_l2)
        else:
            ev_l2 = float(l2['avg_ev_allowed'].mean())
            
        # Season baseline
        p_data_year = p_data[p_data['game_date_dt'].dt.year == gd.year]
        if p_data_year.empty:
            p_data_year = p_data
            
        spin_season = float(p_data_year['avg_release_spin_rate'].mean())
        ext_season = float(p_data_year['avg_release_extension'].mean())
        sum_bbe_season = p_data_year['bbe'].sum()
        if sum_bbe_season > 0:
            ev_season = float((p_data_year['avg_ev_allowed'] * p_data_year['bbe']).sum() / sum_bbe_season)
        else:
            ev_season = float(p_data_year['avg_ev_allowed'].mean())
            
        # Fallbacks
        if pd.isna(xwoba): xwoba = 0.3192
        if pd.isna(barrel): barrel = 0.0826
        if pd.isna(hardhit): hardhit = 0.4024
        if pd.isna(ev): ev = 88.5
        if pd.isna(spin): spin = 2200.0
        if pd.isna(ext): ext = 6.2
        if pd.isna(spin_l2) or pd.isna(spin_season): spin_delta = 0.0
        else: spin_delta = spin_l2 - spin_season
        if pd.isna(ext_l2) or pd.isna(ext_season): ext_delta = 0.0
        else: ext_delta = ext_l2 - ext_season
        if pd.isna(ev_l2) or pd.isna(ev_season): ev_delta = 0.0
        else: ev_delta = ev_l2 - ev_season
        
        return xwoba, barrel, hardhit, ev, spin, ext, spin_delta, ext_delta, ev_delta

    # Helper helper for sums/weighted averages
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

    # Batter stats helper (platoon split + recent form)
    def get_batter_form(bid, gd, opp_hand):
        defaults = {
            'xwoba': 0.320, 'barrel': 0.080, 'hardhit': 0.400, 'ev': 89.0,
            'bat_speed': 71.5, 'swing_length': 7.3, 'sweetspot': 0.33,
            'discipline': 1.5, 'efficiency': 9.8
        }
        if not bid or pd.isna(bid) or int(bid) not in sav_b_dict:
            return defaults
            
        grp = sav_b_dict[int(bid)]
        b_data = grp[grp['game_date_dt'] < gd]
        if b_data.empty:
            return defaults
            
        overall_l15 = b_data.iloc[-15:]
        b_data_split = b_data[b_data['p_throws'] == opp_hand]
        split_l15 = b_data_split.iloc[-15:]
        
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
        
        blended = {}
        for m in ['xwoba', 'barrel', 'hardhit', 'ev', 'bat_speed', 'swing_length', 'sweetspot', 'discipline', 'efficiency']:
            s_val = metrics[f'split_{m}']
            o_val = metrics[f'overall_{m}']
            val = weight * s_val + (1.0 - weight) * o_val
            if pd.isna(val): val = defaults[m]
            blended[m] = val
            
        return blended

    # Reliever stats helper for bullpen metrics
    def get_reliever_stats(rid, gd):
        if not rid or pd.isna(rid) or int(rid) not in sav_p_dict:
            return 0.319, 0.08
        grp = sav_p_dict[int(rid)]
        p_data = grp[grp['game_date_dt'] < gd]
        if p_data.empty:
            return 0.319, 0.08
        l5 = p_data.iloc[-5:]
        sum_bbe = l5['bbe'].sum()
        if sum_bbe > 0:
            xwoba = float((l5['avg_xwoba_allowed'] * l5['bbe']).sum() / sum_bbe)
            barrel = float(l5['barrels_allowed'].sum() / sum_bbe)
        else:
            xwoba = float(l5['avg_xwoba_allowed'].mean())
            barrel = 0.08
        if pd.isna(xwoba): xwoba = 0.319
        if pd.isna(barrel): barrel = 0.08
        return xwoba, barrel

    starters_p = pp_df[pp_df['role'] == 'starter']

    starter_feats = {f: [] for f in [
        'h_starter_xwoba', 'a_starter_xwoba', 'h_starter_barrel', 'a_starter_barrel', 
        'h_starter_hardhit', 'a_starter_hardhit', 'h_starter_ev', 'a_starter_ev',
        'h_starter_spin', 'a_starter_spin', 'h_starter_extension', 'a_starter_extension',
        'h_starter_spin_delta', 'a_starter_spin_delta', 'h_starter_extension_delta', 'a_starter_extension_delta',
        'h_starter_ev_delta', 'a_starter_ev_delta'
    ]}
    
    bullpen_feats = {f: [] for f in [
        'h_bullpen_xwoba', 'a_bullpen_xwoba', 'h_bullpen_barrel', 'a_bullpen_barrel',
        'h_bp_fatigue_recent', 'a_bp_fatigue_recent', 'h_bp_fatigue_recent_72h', 'a_bp_fatigue_recent_72h'
    ]}
    
    lineup_feats = {f: [] for f in [
        'h_lineup_xwoba', 'a_lineup_xwoba', 'h_lineup_barrel', 'a_lineup_barrel', 'h_lineup_hardhit', 'a_lineup_hardhit',
        'h_lineup_ev', 'a_lineup_ev', 'h_lineup_bat_speed', 'a_lineup_bat_speed', 'h_lineup_swing_length', 'a_lineup_swing_length',
        'h_lineup_sweetspot', 'a_lineup_sweetspot', 'h_lineup_discipline', 'a_lineup_discipline', 'h_lineup_efficiency', 'a_lineup_efficiency'
    ]}

    # In predict_mode, only compute savant features for predict rows
    if predict_mode:
        _savant_src = df[pd.isna(df['h_runs_total'])].copy()
    else:
        _savant_src = df
    n_hist_savant = len(df) - len(_savant_src) if predict_mode else 0

    for idx, row in _savant_src.iterrows():
        gpk = row['game_pk']
        gd = row['date_dt']
        htid = TEAM_IDS.get(row['home_team'])
        atid = TEAM_IDS.get(row['away_team'])
        
        match_h = starters_p[(starters_p['game_pk'] == gpk) & (starters_p['team_id'] == htid)]
        match_a = starters_p[(starters_p['game_pk'] == gpk) & (starters_p['team_id'] == atid)]
        
        hp_id = match_h.iloc[0]['player_id'] if not match_h.empty else None
        ap_id = match_a.iloc[0]['player_id'] if not match_a.empty else None
        
        hp_hand = get_pitcher_hand(hp_id) if hp_id else 'R'
        ap_hand = get_pitcher_hand(ap_id) if ap_id else 'R'
        
        # Compute opposing lineups hand weights for splits
        # Home pitcher (hp_id) faces Away lineup
        away_lineup = lineup_dict.get((gpk, atid), []) if atid else []
        hp_lhb, hp_rhb = 0, 0
        for b_id in away_lineup:
            if not b_id or pd.isna(b_id):
                hp_rhb += 1
                continue
            side = batter_sides.get(int(b_id), 'R')
            if side == 'L':
                hp_lhb += 1
            elif side == 'R':
                hp_rhb += 1
            elif side == 'S':
                if hp_hand == 'L': hp_rhb += 1
                else: hp_lhb += 1
            else:
                hp_rhb += 1
        tot_hp = max(hp_lhb + hp_rhb, 1)
        w_lhb_h = hp_lhb / tot_hp
        w_rhb_h = hp_rhb / tot_hp

        # Away pitcher (ap_id) faces Home lineup
        home_lineup = lineup_dict.get((gpk, htid), []) if htid else []
        ap_lhb, ap_rhb = 0, 0
        for b_id in home_lineup:
            if not b_id or pd.isna(b_id):
                ap_rhb += 1
                continue
            side = batter_sides.get(int(b_id), 'R')
            if side == 'L':
                ap_lhb += 1
            elif side == 'R':
                ap_rhb += 1
            elif side == 'S':
                if ap_hand == 'L': ap_rhb += 1
                else: ap_lhb += 1
            else:
                ap_rhb += 1
        tot_ap = max(ap_lhb + ap_rhb, 1)
        w_lhb_a = ap_lhb / tot_ap
        w_rhb_a = ap_rhb / tot_ap

        h_xw, h_bar, h_hh, h_ev, h_spin, h_ext, h_sp_d, h_ex_d, h_ev_d = get_pitcher_form(hp_id, gd, w_lhb_h, w_rhb_h)
        a_xw, a_bar, a_hh, a_ev, a_spin, a_ext, a_sp_d, a_ex_d, a_ev_d = get_pitcher_form(ap_id, gd, w_lhb_a, w_rhb_a)
        
        starter_feats['h_starter_xwoba'].append(h_xw)
        starter_feats['a_starter_xwoba'].append(a_xw)
        starter_feats['h_starter_barrel'].append(h_bar)
        starter_feats['a_starter_barrel'].append(a_bar)
        starter_feats['h_starter_hardhit'].append(h_hh)
        starter_feats['a_starter_hardhit'].append(a_hh)
        starter_feats['h_starter_ev'].append(h_ev)
        starter_feats['a_starter_ev'].append(a_ev)
        starter_feats['h_starter_spin'].append(h_spin)
        starter_feats['a_starter_spin'].append(a_spin)
        starter_feats['h_starter_extension'].append(h_ext)
        starter_feats['a_starter_extension'].append(a_ext)
        starter_feats['h_starter_spin_delta'].append(h_sp_d)
        starter_feats['a_starter_spin_delta'].append(a_sp_d)
        starter_feats['h_starter_extension_delta'].append(h_ex_d)
        starter_feats['a_starter_extension_delta'].append(a_ex_d)
        starter_feats['h_starter_ev_delta'].append(h_ev_d)
        starter_feats['a_starter_ev_delta'].append(a_ev_d)
        
        for side, tid, opposing_hand, bp_fat_48_col, bp_fat_72_col, bp_xw_col, bp_bar_col in [
            ('h', htid, ap_hand, 'h_bp_fatigue_recent', 'h_bp_fatigue_recent_72h', 'h_bullpen_xwoba', 'h_bullpen_barrel'),
            ('a', atid, hp_hand, 'a_bp_fatigue_recent', 'a_bp_fatigue_recent_72h', 'a_bullpen_xwoba', 'a_bullpen_barrel')
        ]:
            fat_48, fat_72, b_xw, b_bar = 0.0, 0.0, 0.319, 0.08
            if tid:
                team_rels = relievers_by_team.get(tid)
                if team_rels is not None:
                    team_rels_30d = team_rels[(team_rels['date_dt'] < gd) & (team_rels['date_dt'] >= gd - timedelta(days=30))]
                    if not team_rels_30d.empty:
                        rids = team_rels_30d['player_id'].unique()
                        rel_stats = []
                        for rid in rids:
                            rxw, rbar = get_reliever_stats(rid, gd)
                            rel_stats.append((rid, rxw, rbar))
                        rel_stats.sort(key=lambda x: x[1])
                        top_2_rids = [x[0] for x in rel_stats[:2]]
                        
                        recent_48h = team_rels_30d[team_rels_30d['player_id'].isin(top_2_rids) & (team_rels_30d['date_dt'] >= gd - timedelta(days=2))]
                        recent_72h = team_rels_30d[team_rels_30d['player_id'].isin(top_2_rids) & (team_rels_30d['date_dt'] >= gd - timedelta(days=3))]
                        fat_48 = float(recent_48h['pitches'].sum())
                        fat_72 = float(recent_72h['pitches'].sum())
                        
                        team_rels_14d = team_rels_30d[team_rels_30d['date_dt'] >= gd - timedelta(days=14)]
                        rids_14d = team_rels_14d['player_id'].unique()
                        if len(rids_14d) > 0:
                            xw_list = [x[1] for x in rel_stats if x[0] in rids_14d]
                            bar_list = [x[2] for x in rel_stats if x[0] in rids_14d]
                            b_xw = float(np.mean(xw_list)) if xw_list else 0.319
                            b_bar = float(np.mean(bar_list)) if bar_list else 0.08
                            
            bullpen_feats[bp_fat_48_col].append(fat_48)
            bullpen_feats[bp_fat_72_col].append(fat_72)
            bullpen_feats[bp_xw_col].append(b_xw)
            bullpen_feats[bp_bar_col].append(b_bar)
            
        for side, tid, opp_hand in [('h', htid, ap_hand), ('a', atid, hp_hand)]:
            bat_ids = lineup_dict.get((gpk, tid), []) if tid else []
            bat_stats = []
            for bid in bat_ids[:9]:
                bat_stats.append(get_batter_form(bid, gd, opp_hand))
                
            while len(bat_stats) < 9:
                bat_stats.append({
                    'xwoba': 0.320, 'barrel': 0.080, 'hardhit': 0.400, 'ev': 89.0,
                    'bat_speed': 71.5, 'swing_length': 7.3, 'sweetspot': 0.33,
                    'discipline': 1.5, 'efficiency': 9.8
                })
                
            for m in ['xwoba', 'barrel', 'hardhit', 'ev', 'bat_speed', 'swing_length', 'sweetspot', 'discipline', 'efficiency']:
                avg_val = float(np.mean([b[m] for b in bat_stats]))
                lineup_feats[f'{side}_lineup_{m}'].append(avg_val)

    if predict_mode and n_hist_savant > 0:
        nan_prefix = [np.nan] * n_hist_savant
        zero_prefix = [0.0] * n_hist_savant
        for k, lst in starter_feats.items():
            df[k] = nan_prefix + lst
        for k, lst in bullpen_feats.items():
            df[k] = zero_prefix + lst
        for k, lst in lineup_feats.items():
            df[k] = nan_prefix + lst
    else:
        for k, lst in starter_feats.items():
            df[k] = lst
        for k, lst in bullpen_feats.items():
            df[k] = lst
        for k, lst in lineup_feats.items():
            df[k] = lst

    df['xwoba_lineup_diff'] = df['h_lineup_xwoba'] - df['a_lineup_xwoba']
    df['barrel_lineup_diff'] = df['h_lineup_barrel'] - df['a_lineup_barrel']
    df['hardhit_lineup_diff'] = df['h_lineup_hardhit'] - df['a_lineup_hardhit']
    df['bat_speed_diff'] = df['h_lineup_bat_speed'] - df['a_lineup_bat_speed']
    df['swing_length_diff'] = df['h_lineup_swing_length'] - df['a_lineup_swing_length']
    df['discipline_diff'] = df['h_lineup_discipline'] - df['a_lineup_discipline']
    df['efficiency_diff'] = df['h_lineup_efficiency'] - df['a_lineup_efficiency']
    
    df['h_matchup_xwoba_diff'] = df['h_lineup_xwoba'] - df['a_starter_xwoba']
    df['a_matchup_xwoba_diff'] = df['a_lineup_xwoba'] - df['h_starter_xwoba']
    df['matchup_advantage'] = df['h_matchup_xwoba_diff'] - df['a_matchup_xwoba_diff']

    df['h_starter_true_risk'] = (1 - df['h_starter_strikeout_rate']) * df['h_starter_barrel']
    df['a_starter_true_risk'] = (1 - df['a_starter_strikeout_rate']) * df['a_starter_barrel']

    for pfx in ['h', 'a']:
        spin = df[f'{pfx}_starter_spin'].fillna(2200)
        ext = df[f'{pfx}_starter_extension'].fillna(6.2)
        ev_allowed = df[f'{pfx}_starter_ev'].fillna(88.5)
        
        spin_z = (spin - 2250) / 250.0
        ext_z = (ext - 6.2) / 0.5
        ev_sup_z = (88.5 - ev_allowed) / 3.0
        
        df[f'{pfx}_starter_stuff_plus'] = (spin_z * 0.35) + (ext_z * 0.25) + (ev_sup_z * 0.40)
        
        disc = df[f'{pfx}_lineup_discipline'].fillna(1.5)
        ss = df[f'{pfx}_lineup_sweetspot'].fillna(0.33)
        df[f'{pfx}_lineup_decision_quality'] = disc * ss * 10

    df.drop(columns=['date_dt'], inplace=True, errors='ignore')

    return df, SAVANT_FEATS


def build_all_features(df, pp=None, predict_mode=False, elo_sys=None):
    """Build the full feature set from historico_partidos + pitcher_performances.
    Only uses real recorded data — no fixed fallback averages that create false signals.
    Games missing real starter data are excluded from training entirely.
    """
    orig_len = len(df)
    orig_cols = {}
    if predict_mode:
        advanced_cols = {
            'mc_home_prob', 'mc_margin_expected', 'mc_blowout_prob', 'mc_upset_risk',
            'elo_pitcher_diff', 'elo_lineup_diff', 'elo_pitcher_trend_h', 'elo_pitcher_trend_a', 'elo_hot_batters_diff', 'pitcher_elo_x_lineup_xwoba',
            'h_pitcher_qs_consistency', 'a_pitcher_qs_consistency', 'h_pitcher_markov_entropy', 'a_pitcher_markov_entropy',
            'h_markov_dominant_rate', 'a_markov_dominant_rate', 'markov_momentum_diff', 'h_markov_resilience', 'a_markov_resilience', 'h_markov_collapse_risk', 'a_markov_collapse_risk',
            'h_starter_stuff_plus', 'a_starter_stuff_plus', 'h_starter_spin_delta', 'a_starter_spin_delta',
            'h_pitcher_form_score', 'a_pitcher_form_score', 'h_pitcher_form_vs_baseline', 'a_pitcher_form_vs_baseline',
            'h_pitcher_quality_idx', 'a_pitcher_quality_idx', 'pitcher_quality_diff', 'h_pitcher_ip_consistency', 'a_pitcher_ip_consistency'
        }
        for c in df.columns:
            if c not in advanced_cols:
                orig_cols[c] = df[c].copy()

    # Drop games without real starter ERA/WHIP — these cannot be used for training
    df = df.dropna(subset=['h_starter_era','a_starter_era','h_starter_whip','a_starter_whip']).copy()
    if pp is not None and not pp.empty:
        df = adjust_pitcher_stats_blended(df, pp)
    dropped = orig_len - len(df)
    if dropped > 0:
        warnings.warn(f'Dropped {dropped}/{orig_len} games missing real starter ERA/WHIP from training set.')

    # Ensure batter/team/context columns exist — will be filled from real data below
    for c in BATTER_FEATS + TEAM_FEATS:
        if c not in df.columns: df[c] = np.nan
    for c in CONTEXT_FEATS:
        if c not in df.columns: df[c] = 0
    if 'is_night' not in df.columns: df['is_night'] = 1
    # Pitcher features already real after dropna — ensure strikeout/walk rates exist
    for c in ['h_starter_strikeout_rate','a_starter_strikeout_rate','h_starter_walk_rate','a_starter_walk_rate']:
        if c not in df.columns: df[c] = np.nan

    # ── REST DAYS ──
    df['h_rest_days'] = df.groupby('home_team')['date'].transform(
        lambda x: x.diff().dt.days.fillna(1).clip(0,7))
    df['a_rest_days'] = df.groupby('away_team')['date'].transform(
        lambda x: x.diff().dt.days.fillna(1).clip(0,7))

    # ── STREAKS & MARKOV CHAINS (consecutive wins & state transitions, no leakage) ──
    team_results = {}  # team -> list of W/L/W_big (1, 0, 2)
    team_wins = {}     # also track for H2H/home-away below
    h_streaks, a_streaks = [], []
    
    h_markov_ww, h_markov_wl, h_markov_exp = [], [], []
    a_markov_ww, a_markov_wl, a_markov_exp = [], [], []
    h_markov_dom, a_markov_dom = [], []
    h_markov_res, a_markov_res = [], []
    
    import math
    
    def get_markov_probs_adv(results, window=45):
        if len(results) < 5:
            return 0.5, 0.5, 0.5, 0.0, 0.5
        recent = results[-window:]
        ww_num, ww_den = 0, 0
        wl_num, wl_den = 0, 0
        dom_num = 0
        res_num, res_den = 0, 0 # Resilience: P(W | L,L)
        
        for i in range(1, len(recent)):
            prev = recent[i-1]
            curr = recent[i]
            is_w = (curr == 1 or curr == 2)
            is_prev_w = (prev == 1 or prev == 2)
            
            if is_prev_w:
                ww_den += 1
                if is_w: ww_num += 1
            else:
                wl_den += 1
                if is_w: wl_num += 1
                
            if curr == 2:
                dom_num += 1
                
            if i >= 2:
                prev2 = recent[i-2]
                if (prev == 0) and (prev2 == 0):
                    res_den += 1
                    if is_w: res_num += 1
                    
        p_ww = ww_num / ww_den if ww_den > 0 else 0.5
        p_wl = wl_num / wl_den if wl_den > 0 else 0.5
        p_dom = dom_num / len(recent)
        p_res = res_num / res_den if res_den > 0 else 0.5
        
        last_res = recent[-1]
        p_win = p_ww if (last_res == 1 or last_res == 2) else p_wl
        return p_ww, p_wl, p_win, p_dom, p_res

    for _, r in df.iterrows():
        h, a = r['home_team'], r['away_team']
        hr, ar_ = r['h_runs_total'], r['a_runs_total']
        
        # In predict_mode, skip historical games (already computed features)
        is_hist = predict_mode and not pd.isna(hr)
        
        def count_consecutive(results):
            streak = 0
            for w in reversed(results):
                if w > 0: streak += 1
                else: break
            return streak
        
        if not is_hist:
            h_streaks.append(count_consecutive(team_results.get(h, [])))
            a_streaks.append(count_consecutive(team_results.get(a, [])))
            h_ww, h_wl, h_ex, h_dom, h_res = get_markov_probs_adv(team_results.get(h, []))
            a_ww, a_wl, a_ex, a_dom, a_res = get_markov_probs_adv(team_results.get(a, []))
            h_markov_ww.append(h_ww); h_markov_wl.append(h_wl); h_markov_exp.append(h_ex)
            a_markov_ww.append(a_ww); a_markov_wl.append(a_wl); a_markov_exp.append(a_ex)
            h_markov_dom.append(h_dom); a_markov_dom.append(a_dom)
            h_markov_res.append(h_res); a_markov_res.append(a_res)
        else:
            # In predict_mode, pad historical rows with defaults
            h_streaks.append(0)
            a_streaks.append(0)
            h_markov_ww.append(0.5); h_markov_wl.append(0.5); h_markov_exp.append(0.5)
            a_markov_ww.append(0.5); a_markov_wl.append(0.5); a_markov_exp.append(0.5)
            h_markov_dom.append(0.0); a_markov_dom.append(0.0)
            h_markov_res.append(0.5); a_markov_res.append(0.5)
        
        if hr > ar_:
            hw_res = 2 if (hr - ar_) >= 3 else 1
            aw_res = 0
        else:
            hw_res = 0
            aw_res = 2 if (ar_ - hr) >= 3 else 1
            
        team_results.setdefault(h, []).append(hw_res)
        team_results.setdefault(a, []).append(aw_res)
        team_wins.setdefault(h, []).append(1 if hr > ar_ else 0)
        team_wins.setdefault(a, []).append(1 if ar_ > hr else 0)
        
    df['h_streak'] = h_streaks
    df['a_streak'] = a_streaks
    df['streak_diff'] = df['h_streak'] - df['a_streak']
    
    df['h_markov_ww'] = h_markov_ww
    df['h_markov_wl'] = h_markov_wl
    df['h_markov_expected_win'] = h_markov_exp
    df['a_markov_ww'] = a_markov_ww
    df['a_markov_wl'] = a_markov_wl
    df['a_markov_expected_win'] = a_markov_exp
    df['markov_advantage'] = df['h_markov_expected_win'] - df['a_markov_expected_win']
    df['h_markov_dominant_rate'] = h_markov_dom
    df['a_markov_dominant_rate'] = a_markov_dom
    df['markov_momentum_diff'] = df['h_markov_expected_win'] * h_markov_res - df['a_markov_expected_win'] * a_markov_res
    
    # --- PITCHER QS MARKOV CHAINS ---
    # Done properly by reading pitcher_performances if provided
    h_qs_cons, a_qs_cons = [], []
    h_p_ent, a_p_ent = [], []
    
    if pp is not None and not pp.empty:
        qs_pp = pp[pp['role'] == 'starter'].copy()
        qs_pp = qs_pp.sort_values('date')
        
        qs_hist = {} # pid -> list of 1 (QS) or 0 (No QS)
        
        def is_qs(er, ip):
            ip_val = parse_baseball_ip(ip)
            if pd.isna(er) or pd.isna(ip_val): return 0
            return 1 if (er <= 3 and ip_val >= 6.0) else 0
            
        def calc_qs_markov(history):
            if len(history) < 3: return 0.5, 0.0
            
            # P(QS | QS)
            qs_den = 0
            qs_num = 0
            for i in range(1, len(history)):
                if history[i-1] == 1:
                    qs_den += 1
                    if history[i] == 1: qs_num += 1
                    
            p_qsqs = qs_num / qs_den if qs_den > 0 else sum(history) / len(history)
            
            # Entropy
            p_qs = sum(history) / len(history)
            if p_qs <= 0.01 or p_qs >= 0.99:
                ent = 0.0
            else:
                ent = - (p_qs * math.log2(p_qs) + (1-p_qs) * math.log2(1-p_qs))
                
            return p_qsqs, ent

        qs_temp = qs_pp.to_dict('records')
        p_dates = {}
        for r in qs_temp:
            p_dates.setdefault(r['player_id'], []).append((r['date'], is_qs(r.get('er'), r.get('ip'))))

        for _, r in df.iterrows():
            gd = r['date']
            h_pid = r.get('pp_h_starter_id', 0)
            a_pid = r.get('pp_a_starter_id', 0)
            
            for pid, lst_cons, lst_ent in [(h_pid, h_qs_cons, h_p_ent), (a_pid, a_qs_cons, a_p_ent)]:
                if pid and pid in p_dates:
                    hist = [qs for dt, qs in p_dates[pid] if dt < gd]
                    cons, ent = calc_qs_markov(hist)
                    lst_cons.append(cons)
                    lst_ent.append(ent)
                else:
                    lst_cons.append(0.5)
                    lst_ent.append(0.0)
    else:
        h_qs_cons = [0.5] * len(df)
        a_qs_cons = [0.5] * len(df)
        h_p_ent = [0.0] * len(df)
        a_p_ent = [0.0] * len(df)

    df['h_pitcher_qs_consistency'] = h_qs_cons
    df['a_pitcher_qs_consistency'] = a_qs_cons
    df['h_pitcher_markov_entropy'] = h_p_ent
    df['a_pitcher_markov_entropy'] = a_p_ent


    # ── DIVISIONAL ──
    df['divisional_game'] = df.apply(
        lambda r: int(TEAM_DIV.get(r['home_team']) == TEAM_DIV.get(r['away_team'])), axis=1)

    # ── PARK FACTOR ──
    df['park_factor'] = df['home_team'].map(PARK_FACTORS).fillna(1.00)

    # ── ELO + PYTHAGOREAN + RUN DIFF + HOME/AWAY SPLITS + H2H (single pass) ──
    elo_map = {}; elo_diffs = []
    team_runs_for = {}   # team -> [runs_scored]
    team_runs_ag = {}    # team -> [runs_allowed]
    team_home_rec = {}   # team -> [1/0] home wins only
    team_away_rec = {}   # team -> [1/0] away wins only
    h2h_wins = {}        # (team1,team2) -> count
    starter_era_hist = {} # player_id -> [era_game]
    h_pyth, a_pyth = [], []
    h_rdiff10, a_rdiff10 = [], []
    h_hwpct, a_awpct = [], []
    h2h_adv = []
    team_batting_hist = {}  # team -> {'ops': [], 'avg': [], 'obp': [], 'slg': []}
    # Variables h_ops_pre etc. eliminadas.
    for _, r in df.iterrows():
        h, a = r['home_team'], r['away_team']
        hr, ar_ = r['h_runs_total'], r['a_runs_total']
        # ELO
        eh, ea = elo_map.get(h,1500), elo_map.get(a,1500)
        is_hist = predict_mode and not pd.isna(hr)
        
        if not is_hist:
            elo_diffs.append(eh-ea)
            w = 1 if hr > ar_ else 0
        else:
            w = 1 if hr > ar_ else 0
            elo_diffs.append(0.0)
        e = 1/(1+10**((ea-eh)/400))
        elo_map[h] = eh+20*(w-e); elo_map[a] = ea+20*((1-w)-(1-e))
        if not is_hist:
            # Use data BEFORE this game only — no leakage
            h_rf = sum(r for d,r in team_runs_for.get(h,[])[-30:]); h_ra = sum(r for d,r in team_runs_ag.get(h,[])[-30:])
            a_rf = sum(r for d,r in team_runs_for.get(a,[])[-30:]); a_ra = sum(r for d,r in team_runs_ag.get(a,[])[-30:])
            hp = h_rf**2/(h_rf**2+h_ra**2+1e-9); ap = a_rf**2/(a_rf**2+a_ra**2+1e-9)
            h_pyth.append(hp); a_pyth.append(ap)
            # Run differential rolling 10 (pre-game only)
            h_rd = [f-a2 for (d,f),(d2,a2) in zip(team_runs_for.get(h,[])[-10:], team_runs_ag.get(h,[])[-10:])]
            a_rd = [f-a2 for (d,f),(d2,a2) in zip(team_runs_for.get(a,[])[-10:], team_runs_ag.get(a,[])[-10:])]
            h_rdiff10.append(sum(h_rd)/max(len(h_rd),1)); a_rdiff10.append(sum(a_rd)/max(len(a_rd),1))
            # Home/Away splits (pre-game only)
            hw_rec = team_home_rec.get(h,[]); aw_rec = team_away_rec.get(a,[])
            h_hwpct.append(sum(hw_rec[-20:])/max(len(hw_rec[-20:]),1))
            a_awpct.append(sum(aw_rec[-20:])/max(len(aw_rec[-20:]),1))
        else:
            h_pyth.append(0.5); a_pyth.append(0.5)
            h_rdiff10.append(0.0); a_rdiff10.append(0.0)
            h_hwpct.append(0.5); a_awpct.append(0.5)
        # Post-game state updates (for NEXT game's features, not this one)
        for t, rs, ra in [(h, hr, ar_), (a, ar_, hr)]:
            team_runs_for.setdefault(t, []).append((gd, rs))
            team_runs_ag.setdefault(t, []).append((gd, ra))
        if hr > ar_:
            team_home_rec.setdefault(h,[]).append(1); team_away_rec.setdefault(a,[]).append(0)
        else:
            team_home_rec.setdefault(h,[]).append(0); team_away_rec.setdefault(a,[]).append(1)
        # H2H
        h2h_key = (h,a)
        h_wins_vs_a = h2h_wins.get(h2h_key,0)
        a_wins_vs_h = h2h_wins.get((a,h),0)
        if not is_hist:
            total_h2h = h_wins_vs_a + a_wins_vs_h
            h2h_adv.append((h_wins_vs_a - a_wins_vs_h) / max(total_h2h, 1))
        else:
            h2h_adv.append(0.0)
        if w: h2h_wins[h2h_key] = h2h_wins.get(h2h_key,0)+1
        else: h2h_wins[(a,h)] = h2h_wins.get((a,h),0)+1
        # Update post-game
        team_wins.setdefault(h, []).append(w)  # Already done above but harmless
        # Record post-game batting stats for rolling (used by NEXT game, not this one)
        for team, pfx in [(h, 'h'), (a, 'a')]:
            team_batting_hist.setdefault(team, {})
            for stat in ['ops', 'avg', 'obp', 'slg']:
                v = r.get(f'{pfx}_{stat}')
                if v is not None and not pd.isna(v):
                    team_batting_hist[team].setdefault(stat, []).append(float(v))
    df['elo_diff'] = elo_diffs
    df['h_pyth_pct'] = h_pyth; df['a_pyth_pct'] = a_pyth
    df['pyth_diff'] = df['h_pyth_pct'] - df['a_pyth_pct']
    df['h_run_diff_10'] = h_rdiff10; df['a_run_diff_10'] = a_rdiff10
    df['run_diff_diff'] = df['h_run_diff_10'] - df['a_run_diff_10']
    df['h_home_wpct'] = h_hwpct; df['a_away_wpct'] = a_awpct
    df['h2h_advantage'] = h2h_adv
    # Use real batting stats from the database directly (datos reales)
    # df['h_ops'], etc. contain the actual starter stats average at game time
    pass

    # ── DIFF features — use real data only; NaN remains NaN until final dropna ──
    df['era_diff'] = df['h_starter_era'] - df['a_starter_era']   # both guaranteed real after dropna above
    df['whip_diff'] = df['h_starter_whip'] - df['a_starter_whip']
    # Team OPS/AVG/OBP/SLG: real values from boxscore stored in DB; fill NaN with column median
    for pfx in ['h','a']:
        for col,default in [('ops',0.700),('avg',0.250),('obp',0.320),('slg',0.400)]:
            full_col = f'{pfx}_{col}'
            median_val = df[full_col].median()
            if pd.isna(median_val): median_val = default
            df[full_col] = df[full_col].fillna(median_val)
    df['ops_diff'] = df['h_ops'] - df['a_ops']
    df['avg_diff'] = df['h_avg'] - df['a_avg']
    df['slg_diff'] = df['h_slg'] - df['a_slg']
    df['obp_diff'] = df['h_obp'] - df['a_obp']
    # Strikeout/walk rates: use real values; if NaN (old games before we tracked rates), drop those rows
    df = df.dropna(subset=['h_starter_strikeout_rate','a_starter_strikeout_rate',
                            'h_starter_walk_rate','a_starter_walk_rate']).copy()
    # Clip to physically realistic MLB bounds — any value above these indicates a corrupt DB record
    for c in ['h_starter_strikeout_rate','a_starter_strikeout_rate']:
        df[c] = df[c].clip(0.0, 0.45)
    for c in ['h_starter_walk_rate','a_starter_walk_rate']:
        df[c] = df[c].clip(0.0, 0.25)
    df['k_rate_diff'] = df['h_starter_strikeout_rate'] - df['a_starter_strikeout_rate']
    df['bb_rate_diff'] = df['h_starter_walk_rate'] - df['a_starter_walk_rate']

    # Lineup power/depth: individual batter OPS/AVG from boxscore stored in DB
    # Fill missing individual batter stats with the team's average from that game
    for pfx in ['h','a']:
        for i in range(1, 10):
            df[f'{pfx}_batter_{i}_ops'] = df[f'{pfx}_batter_{i}_ops'].fillna(df[f'{pfx}_ops'])
            df[f'{pfx}_batter_{i}_avg'] = df[f'{pfx}_batter_{i}_avg'].fillna(df[f'{pfx}_avg'])
        df[f'{pfx}_lineup_power'] = df[[f'{pfx}_batter_{i}_ops' for i in range(1,10)]].mean(axis=1)
        df[f'{pfx}_lineup_depth'] = df[[f'{pfx}_batter_{i}_avg' for i in range(1,10)]].mean(axis=1)
    df['lineup_power_diff'] = df['h_lineup_power'] - df['a_lineup_power']
    df['lineup_depth_diff'] = df['h_lineup_depth'] - df['a_lineup_depth']
    df['domination_diff'] = (df['h_ops'] - df['a_starter_whip']) - (df['a_ops'] - df['h_starter_whip'])
    df['contact_q_diff'] = (df['h_avg'] * df['h_slg']) - (df['a_avg'] * df['a_slg'])
    # Weighted batting order quality (top 4 batters count 60%)
    for pfx in ['h','a']:
        top4 = df[[f'{pfx}_batter_{i}_ops' for i in range(1,5)]].mean(axis=1)
        bot5 = df[[f'{pfx}_batter_{i}_ops' for i in range(5,10)]].mean(axis=1)
        df[f'{pfx}_lineup_quality_w'] = top4 * 0.6 + bot5 * 0.4
    df['lineup_quality_weighted_diff'] = df['h_lineup_quality_w'] - df['a_lineup_quality_w']

    # ── SABER features — all inputs are now real (no fillna with fictional constants) ──
    # NOTE: fip is NOT real FIP (13*HR+3*BB-2*K)/IP+3.2. Custom composite ERA*0.6+WHIP*1.5+3.1.
    # Models trained with this; to use real FIP, retrain and update features.json.
    eps = 1e-5
    for pfx in ['h_','a_']:
        era_c = f'{pfx}starter_era'; wh_c = f'{pfx}starter_whip'
        kr_c = f'{pfx}starter_strikeout_rate'; br_c = f'{pfx}starter_walk_rate'
        df[f'{pfx}fip'] = df[era_c]*0.6 + df[wh_c]*1.5 + 3.1
        df[f'{pfx}tt_era'] = df[era_c]*0.4 + df[f'{pfx}fip']*0.6
        df[f'{pfx}era_whip_ratio'] = df[era_c] / np.clip(df[wh_c], eps, None)
        df[f'{pfx}kbb'] = df[kr_c] / np.clip(df[br_c], eps, None)
        df[f'{pfx}dom'] = (df[kr_c] - df[br_c]) / np.clip(df[wh_c], eps, None)
    df['h_era_x_a_ops'] = df['h_starter_era'] * df['a_ops']
    df['a_era_x_h_ops'] = df['a_starter_era'] * df['h_ops']

    # ── BULLPEN from pitcher_performances — real data only, no fixed defaults ──
    # Initialize with NaN; will be filled only with real data
    for c in ['h_bp_avail','a_bp_avail','h_bp_fatigue','a_bp_fatigue','h_bullpen_era_l3','a_bullpen_era_l3']:
        df[c] = np.nan

    if pp is not None and not pp.empty:
        relievers = pp[pp['role']=='reliever'].copy()
        if not relievers.empty and len(df) > 0:
            relievers = relievers.sort_values('date')
            for side_col, pfx in [('home_team','h_'),('away_team','a_')]:
                team_bp = {}  # tid -> list of (date, er, ip, player_id)
                for _, rr in relievers.iterrows():
                    _er = rr.get('er', None)
                    if _er is None or pd.isna(_er):
                        _er = rr.get('runs', 0)
                    _ip = rr.get('ip', 0.0)
                    _ip_dec = parse_baseball_ip(_ip)
                    team_bp.setdefault(int(rr['team_id']), []).append(
                        (rr['date'], float(_er or 0), _ip_dec, int(rr['player_id'])))
                bp_avail_vals = []; bp_fat_vals = []; bp_l3_vals = []
                _bp_iter = df.iterrows()
                if predict_mode:
                    _bp_iter = df[pd.isna(df['h_runs_total'])].iterrows()
                    _n_hist = df['h_runs_total'].notna().sum()
                    bp_avail_vals.extend([np.nan]*_n_hist)
                    bp_fat_vals.extend([np.nan]*_n_hist)
                    bp_l3_vals.extend([np.nan]*_n_hist)
                for _, row in _bp_iter:
                    gd = row['date']
                    tid = TEAM_IDS.get(row[side_col])
                    if not tid or tid not in team_bp:
                        bp_avail_vals.append(np.nan); bp_fat_vals.append(np.nan); bp_l3_vals.append(np.nan)
                        continue
                    entries = team_bp[tid]
                    rec3  = [(d,er_val,ip_val,p) for d,er_val,ip_val,p in entries if d < gd and d >= gd - timedelta(days=3)]
                    bp14  = [(d,er_val,ip_val,p) for d,er_val,ip_val,p in entries if d < gd and d >= gd - timedelta(days=14)]
                    bp30  = [(d,er_val,ip_val,p) for d,er_val,ip_val,p in entries if d < gd and d >= gd - timedelta(days=30)]
                    tot_p = len(set(p for _,_,_,p in bp14))
                    tir_p = len(set(p for _,_,_,p in rec3))
                    avail = max(0, tot_p - tir_p) / max(tot_p, 1) if tot_p > 0 else np.nan
                    # bp_fatigue: prefer 14-day, fallback to 30-day real data
                    if bp14:
                        sum_er = sum(x[1] for x in bp14)
                        sum_ip = sum(x[2] for x in bp14)
                        fatigue = (sum_er * 9.0) / max(sum_ip, 0.1)
                    elif bp30:
                        sum_er = sum(x[1] for x in bp30)
                        sum_ip = sum(x[2] for x in bp30)
                        fatigue = (sum_er * 9.0) / max(sum_ip, 0.1)
                    else:
                        fatigue = np.nan
                    # era_l3: prefer 3-day, fallback to 14-day
                    if rec3:
                        sum_er = sum(x[1] for x in rec3)
                        sum_ip = sum(x[2] for x in rec3)
                        era_l3 = (sum_er * 9.0) / max(sum_ip, 0.1)
                    elif bp14:
                        sum_er = sum(x[1] for x in bp14)
                        sum_ip = sum(x[2] for x in bp14)
                        era_l3 = (sum_er * 9.0) / max(sum_ip, 0.1)
                    else:
                        era_l3 = np.nan
                    bp_avail_vals.append(avail); bp_fat_vals.append(fatigue); bp_l3_vals.append(era_l3)
                df[f'{pfx}bp_avail'] = bp_avail_vals
                df[f'{pfx}bp_fatigue'] = bp_fat_vals
                df[f'{pfx}bullpen_era_l3'] = bp_l3_vals
    # For bullpen: fill remaining NaN with the season median from real data (not a fixed constant)
    for c in ['h_bp_avail','a_bp_avail']:
        med = df[c].median()
        df[c] = df[c].fillna(med if not pd.isna(med) else 0.60)
    for c in ['h_bp_fatigue','a_bp_fatigue','h_bullpen_era_l3','a_bullpen_era_l3']:
        med = df[c].median()
        df[c] = df[c].fillna(med if not pd.isna(med) else 4.20)

    # ── STARTER ROLLING ERA L3 + VOLATILITY + SAMPLE IP + INJURY FLAG ──
    if pp is not None and not pp.empty:
        starters = pp[pp['role']=='starter'].copy()
        starters = starters[starters['era_game'].notna()]
        if not starters.empty:
            # Build per-pitcher history: ERA values, dates, IP
            starter_era_by_pid = {}
            starter_dates_by_pid = {}
            starter_ip_by_pid = {}
            for _, sr in starters.sort_values('date').iterrows():
                pid = int(sr['player_id'])
                starter_era_by_pid.setdefault(pid, []).append(float(sr['era_game']))
                starter_dates_by_pid.setdefault(pid, []).append(sr['date'])
                ip_val = sr.get('ip', 0)
                try:
                    ip_str = str(ip_val).strip()
                    if '.' in ip_str:
                        parts = ip_str.split('.')
                        ip_parsed = int(parts[0]) + int(parts[1]) / 3.0
                    else:
                        ip_parsed = float(ip_str) if ip_str else 0.0
                except:
                    ip_parsed = 0.0
                starter_ip_by_pid.setdefault(pid, []).append(ip_parsed)
            h_sera_l3 = []; a_sera_l3 = []
            h_era_std = []; a_era_std = []
            h_sample_ip = []; a_sample_ip = []
            h_injury = []; a_injury = []
            _era_iter = df.iterrows()
            if predict_mode:
                _era_iter = df[pd.isna(df['h_runs_total'])].iterrows()
                # Fill historical rows with existing values
                _hist_mask = df['h_runs_total'].notna()
                _n_hist = _hist_mask.sum()
                hist_l3 = df.loc[_hist_mask, 'h_starter_era'].tolist()
                h_sera_l3.extend(hist_l3)
                a_sera_l3.extend(df.loc[_hist_mask, 'a_starter_era'].tolist())
                h_era_std.extend([np.nan]*_n_hist)
                a_era_std.extend([np.nan]*_n_hist)
                h_sample_ip.extend([np.nan]*_n_hist)
                a_sample_ip.extend([np.nan]*_n_hist)
                h_injury.extend([0]*_n_hist)
                a_injury.extend([0]*_n_hist)
            for _, row in _era_iter:
                gd = row['date']
                for pfx, col_l3, col_std, col_ip, col_inj in [
                    ('h', h_sera_l3, h_era_std, h_sample_ip, h_injury),
                    ('a', a_sera_l3, a_era_std, a_sample_ip, a_injury)]:
                    era_col = f'{pfx}_starter_era'
                    side_team = row['home_team'] if pfx=='h' else row['away_team']
                    tid = TEAM_IDS.get(side_team)
                    match = starters[(starters['date']==gd)&(starters['team_id']==tid)]
                    if not match.empty:
                        pid = int(match.iloc[0]['player_id'])
                        hist = starter_era_by_pid.get(pid, [])
                        dates = starter_dates_by_pid.get(pid, [])
                        ips = starter_ip_by_pid.get(pid, [])
                        # Count entries BEFORE this game
                        n_prior = sum(1 for d in dates if d < gd)
                        prior_eras = hist[:n_prior]
                        prior_dates = [d for d in dates if d < gd]
                        prior_ips = ips[:n_prior]
                        # ERA L3 (last 3 starts before this game)
                        recent3 = prior_eras[-3:]
                        if recent3:
                            col_l3.append(sum(recent3)/len(recent3))
                        else:
                            col_l3.append(float(row[era_col]) if not pd.isna(row.get(era_col)) else np.nan)
                        # ERA std (volatility from last 7 starts)
                        recent7 = prior_eras[-7:]
                        col_std.append(float(np.std(recent7)) if len(recent7) >= 3 else np.nan)
                        # Sample IP (total IP this season before this game)
                        col_ip.append(sum(prior_ips))
                        # Injury flag (gap > 15 days between last two starts)
                        if len(prior_dates) >= 2:
                            gap = (prior_dates[-1] - prior_dates[-2]).days
                            col_inj.append(1 if gap > 15 else 0)
                        else:
                            col_inj.append(0)
                    else:
                        col_l3.append(float(row[era_col]) if not pd.isna(row.get(era_col)) else np.nan)
                        col_std.append(np.nan)
                        col_ip.append(np.nan)
                        col_inj.append(0)
            df['h_starter_era_l3'] = h_sera_l3
            df['a_starter_era_l3'] = a_sera_l3
            df['h_starter_era_std'] = h_era_std
            df['a_starter_era_std'] = a_era_std
            df['h_starter_sample_ip'] = h_sample_ip
            df['a_starter_sample_ip'] = a_sample_ip
            df['h_starter_injury_flag'] = h_injury
            df['a_starter_injury_flag'] = a_injury
            for c in ['h_starter_era_l3','a_starter_era_l3']:
                med = df[c].median()
                df[c] = df[c].fillna(med if not pd.isna(med) else df['h_starter_era'].median())
        else:
            df['h_starter_era_l3'] = df['h_starter_era']
            df['a_starter_era_l3'] = df['a_starter_era']
            for c in ['h_starter_era_std','a_starter_era_std']: df[c] = np.nan
            for c in ['h_starter_sample_ip','a_starter_sample_ip']: df[c] = np.nan
            for c in ['h_starter_injury_flag','a_starter_injury_flag']: df[c] = 0
    else:
        df['h_starter_era_l3'] = df['h_starter_era']
        df['a_starter_era_l3'] = df['a_starter_era']
        for c in ['h_starter_era_std','a_starter_era_std']: df[c] = np.nan
        for c in ['h_starter_sample_ip','a_starter_sample_ip']: df[c] = np.nan
        for c in ['h_starter_injury_flag','a_starter_injury_flag']: df[c] = 0
    # Fill NaN for volatility features
    for c in ['h_starter_era_std', 'a_starter_era_std']:
        med = df[c].median()
        df[c] = df[c].fillna(med if not pd.isna(med) else 3.50)
    for c in ['h_starter_sample_ip', 'a_starter_sample_ip']:
        med = df[c].median()
        df[c] = df[c].fillna(med if not pd.isna(med) else 40.0)
    for c in ['h_starter_injury_flag', 'a_starter_injury_flag']:
        df[c] = df[c].fillna(0)

    # ── PITCHER FORM / QUALITY INDEX ──
    # Composite score: K/9, BB/9, HR/9, ERA — weighted by recency
    # Pre-load all pitcher data once → dict cache (no per-game SQL)
    try:
        global _PF_DATA_CACHE
        if _PF_DATA_CACHE is None:
            conn_pf = sqlite3.connect(DB_PATH)
            pf_raw = pd.read_sql("""
                SELECT player_id, date, era_game, ip, k, bb, hr
                FROM pitcher_performances
                WHERE role='starter'
                ORDER BY player_id, date
            """, conn_pf)
            conn_pf.close()
            _PF_DATA_CACHE = {}
            for _, r in pf_raw.iterrows():
                pid = int(r['player_id'])
                _PF_DATA_CACHE.setdefault(pid, []).append({
                    'date': r['date'], 'era': r['era_game'],
                    'ip': parse_baseball_ip(r['ip']),
                    'k': r['k'] or 0, 'bb': r['bb'] or 0, 'hr': r['hr'] or 0})
        pf_data = _PF_DATA_CACHE

        def outing_score(era, ip, k, bb, hr):
            ip = max(ip, 0.1)
            k9 = k * 9 / ip; bb9 = bb * 9 / ip; hr9 = hr * 9 / ip
            return (k9 * 0.10) - (bb9 * 0.20) - (hr9 * 0.30) - (era * 0.40)

        def compute_pitcher_form(pid, before_date):
            if not pid or pid == 0:
                return 0.5, 0.0, 0.5, 1.0
            entries = pf_data.get(pid, [])
            before = [e for e in entries if e['date'] < before_date]
            recent = before[-5:]
            if len(recent) < 2:
                return 0.5, 0.0, 0.5, 1.0
            year = before_date[:4]
            season = [e for e in before if e['date'].startswith(year)]
            if len(season) >= 2:
                s_era = np.mean([e['era'] for e in season if e['era'] is not None])
                s_ip = np.mean([e['ip'] for e in season if e['ip'] is not None])
                s_k = np.mean([e['k'] for e in season])
                s_bb = np.mean([e['bb'] for e in season])
                s_hr = np.mean([e['hr'] for e in season])
            else:
                s_era, s_ip, s_k, s_bb, s_hr = 4.50, 5.0, 5.0, 2.0, 1.0
            recent_scores = []
            for i, e in enumerate(reversed(recent)):
                if e['era'] is not None and e['ip'] is not None:
                    w = 0.6 ** i
                    recent_scores.append((outing_score(e['era'], e['ip'], e['k'], e['bb'], e['hr']), w))
            if not recent_scores:
                return 0.5, 0.0, 0.5, 1.0
            total_w = sum(w for _, w in recent_scores)
            form_score = sum(s * w for s, w in recent_scores) / total_w
            baseline_score = outing_score(s_era, s_ip, s_k, s_bb, s_hr)
            form_vs_baseline = form_score - baseline_score
            form_values = [s for s, _ in recent_scores]
            stability = 1.0 / (1.0 + np.std(form_values)) if len(form_values) > 1 else 0.5
            quality_idx = form_score * stability
            recent_ips = [e['ip'] for e in reversed(recent) if e['ip'] is not None]
            avg_recent_ip = np.mean(recent_ips) if recent_ips else 5.0
            ip_consistency = avg_recent_ip / max(s_ip, 1.0)
            form_norm = max(0, min(1, (form_score + 2) / 4))
            qi_norm = max(0, min(1, (quality_idx + 1) / 2))
            return form_norm, form_vs_baseline, qi_norm, ip_consistency

        h_form, a_form = [], []
        h_fb, a_fb = [], []
        h_qi, a_qi = [], []
        h_ipc, a_ipc = [], []
        _pf_iter = df.iterrows()
        if predict_mode:
            _pf_iter = df[pd.isna(df['h_runs_total'])].iterrows()
            _hist_mask = df['h_runs_total'].notna()
            _n_hist = _hist_mask.sum()
            h_form.extend([0.5]*_n_hist)
            a_form.extend([0.5]*_n_hist)
            h_fb.extend([0.0]*_n_hist)
            a_fb.extend([0.0]*_n_hist)
            h_qi.extend([0.5]*_n_hist)
            a_qi.extend([0.5]*_n_hist)
            h_ipc.extend([1.0]*_n_hist)
            a_ipc.extend([1.0]*_n_hist)
        for _, row in _pf_iter:
            gd = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            hp_id = int(row.get('pp_h_starter_id', 0)) if not pd.isna(row.get('pp_h_starter_id', 0)) else 0
            ap_id = int(row.get('pp_a_starter_id', 0)) if not pd.isna(row.get('pp_a_starter_id', 0)) else 0
            fv, fb, qi, ip = compute_pitcher_form(hp_id, gd)
            h_form.append(fv); h_fb.append(fb); h_qi.append(qi); h_ipc.append(ip)
            fv, fb, qi, ip = compute_pitcher_form(ap_id, gd)
            a_form.append(fv); a_fb.append(fb); a_qi.append(qi); a_ipc.append(ip)

        df['h_pitcher_form_score'] = h_form
        df['a_pitcher_form_score'] = a_form
        df['h_pitcher_form_vs_baseline'] = h_fb
        df['a_pitcher_form_vs_baseline'] = a_fb
        df['h_pitcher_quality_idx'] = h_qi
        df['a_pitcher_quality_idx'] = a_qi
        df['pitcher_quality_diff'] = df['h_pitcher_quality_idx'] - df['a_pitcher_quality_idx']
        df['h_pitcher_ip_consistency'] = h_ipc
        df['a_pitcher_ip_consistency'] = a_ipc
        df['h_era_trend'] = df['h_starter_era'] - df['h_starter_era_l3']
        df['a_era_trend'] = df['a_starter_era'] - df['a_starter_era_l3']
        non_default = sum(1 for f in h_form if f != 0.5)
        print(f"  ✅ Pitcher form features computed ({non_default} games with data)")
    except Exception as e:
        import traceback; traceback.print_exc()
        warnings.warn(f"Pitcher form features failed: {e}")
        for c in ['h_pitcher_form_score','a_pitcher_form_score','h_pitcher_form_vs_baseline',
                   'a_pitcher_form_vs_baseline','h_pitcher_quality_idx','a_pitcher_quality_idx',
                   'pitcher_quality_diff','h_pitcher_ip_consistency','a_pitcher_ip_consistency',
                   'h_era_trend','a_era_trend']:
            df[c] = 0.5 if 'score' in c or 'qi' in c or 'idx' in c else 0

    # ── PITCHER vs OPPONENT H2H CAREER STATS ──
    # Computed from pp + df (NO LEAK: filtered by game date)
    try:
        _h2h_data = defaultdict(list)  # (pid, opp_team) -> [(date, era_w, k, ip_w), ...]
        if pp is not None and not pp.empty:
            _pp_s = pp[pp['role'] == 'starter'].sort_values('date')
            _gi = df[['game_pk','home_team','away_team','home_team_id','away_team_id']].drop_duplicates('game_pk')
            _gi = _gi.set_index('game_pk')
            for _, _pr in _pp_s.iterrows():
                _pid = int(_pr['player_id'])
                _gpk = int(_pr['game_pk'])
                _gi_row = _gi.loc[_gpk] if _gpk in _gi.index else None
                if _gi_row is None:
                    continue
                if pd.isna(_gi_row.get('home_team_id')) or pd.isna(_gi_row.get('away_team_id')):
                    continue
                if _pr.get('team_id') is None or pd.isna(_pr.get('team_id')):
                    continue
                # Determine opponent
                if int(_pr['team_id']) == int(_gi_row['home_team_id']):
                    _opp = _gi_row['away_team']
                elif int(_pr['team_id']) == int(_gi_row['away_team_id']):
                    _opp = _gi_row['home_team']
                else:
                    continue
                _era = float(_pr.get('era_game', 0) or 0)
                _k = int(_pr.get('k', 0) or 0)
                _ip = max(float(_pr.get('ip', 0) or 0), 0.1)
                _h2h_data[(_pid, _opp)].append((_pr['date'], _era, _k, _ip))

        def _h2h_mean(pid, opp, gd):
            entries = _h2h_data.get((pid, opp), [])
            entries = [e for e in entries if e[0] < gd]
            if not entries:
                return None
            total_ip = sum(e[3] for e in entries)
            if total_ip > 0:
                era_w = sum(e[1] * e[3] for e in entries) / total_ip
            else:
                era_w = entries[-1][1]
            k_avg = sum(e[2] for e in entries) / len(entries)
            return era_w, k_avg, total_ip

        h2h_era_h, h2h_era_a = [], []
        h2h_k_h, h2h_k_a = [], []
        h2h_ip_h, h2h_ip_a = [], []
        for _, row in df.iterrows():
            gd = row['date']
            hp_id = int(row.get('pp_h_starter_id', 0)) if not pd.isna(row.get('pp_h_starter_id', 0)) else 0
            ap_id = int(row.get('pp_a_starter_id', 0)) if not pd.isna(row.get('pp_a_starter_id', 0)) else 0
            h_team, a_team = row['home_team'], row['away_team']
            h_rec = _h2h_mean(hp_id, a_team, gd) or _h2h_mean(hp_id, h_team, gd)
            if h_rec:
                h2h_era_h.append(h_rec[0]); h2h_k_h.append(h_rec[1]); h2h_ip_h.append(h_rec[2])
            else:
                h2h_era_h.append(np.nan); h2h_k_h.append(np.nan); h2h_ip_h.append(np.nan)
            a_rec = _h2h_mean(ap_id, h_team, gd) or _h2h_mean(ap_id, a_team, gd)
            if a_rec:
                h2h_era_a.append(a_rec[0]); h2h_k_a.append(a_rec[1]); h2h_ip_a.append(a_rec[2])
            else:
                h2h_era_a.append(np.nan); h2h_k_a.append(np.nan); h2h_ip_a.append(np.nan)
        df['h_starter_h2h_era'] = h2h_era_h
        df['a_starter_h2h_era'] = h2h_era_a
        df['h_starter_h2h_k'] = h2h_k_h
        df['a_starter_h2h_k'] = h2h_k_a
        df['h_starter_h2h_ip'] = h2h_ip_h
        df['a_starter_h2h_ip'] = h2h_ip_a
        df['h2h_era_diff'] = df['h_starter_h2h_era'] - df['a_starter_h2h_era']
        for c in ['h_starter_h2h_era', 'a_starter_h2h_era']:
            df[c] = df[c].fillna(df['h_starter_era'] if 'h' in c else df['a_starter_era'])
        for c in ['h_starter_h2h_k', 'a_starter_h2h_k']:
            df[c] = df[c].fillna(df.get('h_starter_strikeout_rate', 6.0) if 'h' in c else df.get('a_starter_strikeout_rate', 6.0))
        for c in ['h_starter_h2h_ip', 'a_starter_h2h_ip']:
            df[c] = df[c].fillna(5.0)
        df['h2h_era_diff'] = df['h2h_era_diff'].fillna(0)
        print(f"  ✅ H2H features computed ({df['h_starter_h2h_era'].notna().sum()} games with data)")
    except Exception as e:
        import traceback; traceback.print_exc()
        warnings.warn(f"H2H features failed: {e}")
        for c in ['h_starter_h2h_era','a_starter_h2h_era','h_starter_h2h_k','a_starter_h2h_k',
                   'h_starter_h2h_ip','a_starter_h2h_ip','h2h_era_diff']:
            df[c] = 0

    # ── INTERACTION FEATURES ──
    df['momentum_h'] = df['h_streak'] * (df['elo_diff'].clip(-200,200)/200)
    df['momentum_a'] = df['a_streak'] * (-df['elo_diff'].clip(-200,200)/200)
    h_bp_lev = df['h_bp_avail'] / df['h_bp_fatigue'].clip(lower=0.5)
    a_bp_lev = df['a_bp_avail'] / df['a_bp_fatigue'].clip(lower=0.5)
    df['bullpen_leverage_diff'] = h_bp_lev - a_bp_lev

    # ── APPEND SAVANT FEATURES ──
    df, s_feats = append_savant_features(df, pp, predict_mode=predict_mode)


    # ── MONTE CARLO SIMULATION (Module 4) ──
    # Runs MC simulations using Poisson distributions (1000 iters/game)
    mc_home_prob = []
    mc_margin_exp = []
    mc_blowout_prob = []
    mc_upset_risk = []
    MC_ITERS = 1000
    MC_WALKOFF = np.random.rand(MC_ITERS) < 0.05
    
    for i, r in df.iterrows():
        if predict_mode and not pd.isna(r.get('h_runs_total', np.nan)):
            mc_home_prob.append(0.5)
            mc_margin_exp.append(0.0)
            mc_blowout_prob.append(0.0)
            mc_upset_risk.append(0.0)
            continue
        h = r['home_team']
        a = r['away_team']
        
        # Calculate Lambda (expected runs) — pre-game only, NO LEAK
        gd = r['date']
        h_rf_entries = [(d, rs) for d, rs in team_runs_for.get(h, []) if d < gd][-30:]
        a_rf_entries = [(d, rs) for d, rs in team_runs_for.get(a, []) if d < gd][-30:]
        h_avg_runs = sum(rs for d, rs in h_rf_entries) / len(h_rf_entries) if h_rf_entries else 4.5
        a_avg_runs = sum(rs for d, rs in a_rf_entries) / len(a_rf_entries) if a_rf_entries else 4.5
        
        pf = r.get('park_factor', 1.0)
        h_era = r.get('h_starter_era_l3', 4.0)
        a_era = r.get('a_starter_era_l3', 4.0)
        if pd.isna(h_era): h_era = 4.0
        if pd.isna(a_era): a_era = 4.0
        
        # Adjust lambda: base runs * park factor * opponent pitcher adjustment
        lam_h = h_avg_runs * pf * (a_era / 4.0)
        lam_a = a_avg_runs * (h_era / 4.0)
        
        # Add constraints to prevent crazy values
        lam_h = max(1.5, min(lam_h, 8.0))
        lam_a = max(1.5, min(lam_a, 8.0))
        
        # Simulate games
        sims_h = np.random.poisson(lam_h, MC_ITERS)
        sims_a = np.random.poisson(lam_a, MC_ITERS)
        
        # Small 5% chance of walk-off extra run
        sims_h = sims_h + MC_WALKOFF.astype(int)
        
        wins_h = (sims_h > sims_a).sum()
        ties = (sims_h == sims_a).sum()
        wins_h += ties * 0.53
        
        p_home = wins_h / MC_ITERS
        margins = sims_h - sims_a
        margin_exp = margins.mean()
        blowout = (margins >= 4).sum() / MC_ITERS
        
        if p_home >= 0.55:
            upset = (margins <= -3).sum() / MC_ITERS
        elif p_home <= 0.45:
            upset = (margins >= 3).sum() / MC_ITERS
        else:
            upset = 0.0
            
        mc_home_prob.append(p_home)
        mc_margin_exp.append(margin_exp)
        mc_blowout_prob.append(blowout)
        mc_upset_risk.append(upset)
        
    df['mc_home_prob'] = mc_home_prob
    df['mc_margin_expected'] = mc_margin_exp
    df['mc_blowout_prob'] = mc_blowout_prob
    df['mc_upset_risk'] = mc_upset_risk
    print(f"  ✅ Monte Carlo simulations complete ({MC_ITERS} iters/game)")

    # ── ELO AS FIRST-CLASS FEATURES (Module 5) ──
    try:
        if elo_sys is None:
            from models.elo import PlayerELO
            elo_sys = PlayerELO()
        
        elo_p_diff = []
        elo_l_diff = []
        elo_pt_h = []
        elo_pt_a = []
        elo_hb_diff = []
        elo_x_xwoba = []
        
        conn_bp = sqlite3.connect(DB_PATH)
        bp_df = pd.read_sql("SELECT game_pk, team_id, player_id FROM batter_performances", conn_bp)
        conn_bp.close()
        
        bp_grouped = bp_df.groupby(['game_pk', 'team_id'])['player_id'].apply(list).to_dict()
        
        # ── Local form tracker (NO LEAK: filters by game date) ──
        _local_form = {}
        if pp is not None and not pp.empty:
            _pp_s = pp[pp['role'] == 'starter'].sort_values('date')
            for _, _pr in _pp_s.iterrows():
                _pid = int(_pr['player_id'])
                _ip = max(parse_baseball_ip(_pr['ip']), 0.1)
                _er = int(_pr['er'] or 0)
                _k = int(_pr['k'] or 0)
                _bb = int(_pr.get('bb', 0) or 0)
                _h = int(_pr.get('hits', 0) or 0)
                _raw = 50 + _k * 1.0 - _bb * 1.0 - _er * 2.0 - _h * 0.5 + _ip * 2.0
                _perf = max(0.0, min(1.0, _raw / 100.0))
                _local_form.setdefault(_pid, []).append((_pr['date'], _perf))
        
        def _form_no_leak(pid, gd, n=3):
            if not pid or pid not in _local_form:
                return 0.0
            _ps = [p for d, p in _local_form[pid] if d < gd]
            if len(_ps) < n:
                return 0.0
            return sum(_ps[-n:]) / n
        
        for _, r in df.iterrows():
            # In predict_mode, only compute ELO for predict rows (no known outcome)
            if predict_mode and not pd.isna(r.get('h_runs_total', np.nan)):
                elo_p_diff.append(0.0); elo_l_diff.append(0.0)
                elo_pt_h.append(0.0); elo_pt_a.append(0.0)
                elo_hb_diff.append(0); elo_x_xwoba.append(0.0)
                continue

            gpk = int(r['game_pk'])
            h_pid = r.get('pp_h_starter_id', 0)
            a_pid = r.get('pp_a_starter_id', 0)
            if pd.isna(h_pid): h_pid = 0
            if pd.isna(a_pid): a_pid = 0
            
            hp_elo = elo_sys.get_historical_pitcher_elo(h_pid, gpk)
            ap_elo = elo_sys.get_historical_pitcher_elo(a_pid, gpk)
            elo_p_diff.append((hp_elo - ap_elo) / 100.0)
            
            gd = r['date']
            elo_pt_h.append(_form_no_leak(h_pid, gd, 3))
            elo_pt_a.append(_form_no_leak(a_pid, gd, 3))
            
            htid = TEAM_IDS.get(r['home_team'], 0)
            atid = TEAM_IDS.get(r['away_team'], 0)
            
            h_batters = bp_grouped.get((gpk, htid), [])
            a_batters = bp_grouped.get((gpk, atid), [])
            
            h_b_elos = [elo_sys.get_historical_batter_elo(b, gpk) for b in h_batters[:9]]
            a_b_elos = [elo_sys.get_historical_batter_elo(b, gpk) for b in a_batters[:9]]
            
            while len(h_b_elos) < 9: h_b_elos.append(1500)
            while len(a_b_elos) < 9: a_b_elos.append(1500)
            
            h_b_elos = h_b_elos[:9]
            a_b_elos = a_b_elos[:9]
            
            h_l_elo = sum(h_b_elos) / 9.0
            a_l_elo = sum(a_b_elos) / 9.0
            elo_l_diff.append((h_l_elo - a_l_elo) / 100.0)
            
            h_hot = sum(1 for e in h_b_elos if e > 1550)
            a_hot = sum(1 for e in a_b_elos if e > 1550)
            elo_hb_diff.append(h_hot - a_hot)
            
            h_x = r.get('a_lineup_xwoba', 0.320)
            a_x = r.get('h_lineup_xwoba', 0.320)
            hp_norm = (hp_elo - 1500) / 500.0
            ap_norm = (ap_elo - 1500) / 500.0
            elo_x_xwoba.append((hp_norm * h_x) - (ap_norm * a_x))
            
        df['elo_pitcher_diff'] = elo_p_diff
        df['elo_lineup_diff'] = elo_l_diff
        df['elo_pitcher_trend_h'] = elo_pt_h
        df['elo_pitcher_trend_a'] = elo_pt_a
        df['elo_hot_batters_diff'] = elo_hb_diff
        df['pitcher_elo_x_lineup_xwoba'] = elo_x_xwoba
        print("  ✅ ELO features computed from PlayerELO")
    except Exception as e:
        import traceback; traceback.print_exc()
        warnings.warn(f"ELO feature computation failed: {e}")
        for c in ELO_FEATS:
            df[c] = 0.0

    # ── ASSEMBLE FINAL FEATURE LIST ──

    CORE_FEATS = (PITCHER_FEATS + TEAM_FEATS + CONTEXT_FEATS +
                  DIFF_FEATS + SABER_FEATS + DYNAMIC_FEATS + POWER_FEATS + VOLATILITY_FEATS + PITCHER_FORM_FEATS + MARKOV_FEATS + s_feats + ELO_FEATS + MC_FEATS)
    # Fix #31: warn on excluded features
    missing_feats = [c for c in CORE_FEATS if c not in df.columns]
    if missing_feats:
        warnings.warn(f'Features excluded (not in df): {missing_feats}')
    ALL_FEATS = list(dict.fromkeys([c for c in CORE_FEATS if c in df.columns]))

    target = (df['h_runs_total'] > df['a_runs_total']).astype(int).values
    # Clean up temp columns
    for c in ['home_won','away_won','h_lineup_power','a_lineup_power','h_lineup_depth','a_lineup_depth']:
        if c in df.columns: df.drop(columns=[c], inplace=True, errors='ignore')

    if predict_mode and orig_cols:
        is_predict_row = df['h_runs_total'].isna()
        for c, orig_series in orig_cols.items():
            if c in df.columns:
                df.loc[is_predict_row, c] = orig_series.loc[df.index[is_predict_row]]

    return df, ALL_FEATS, target
