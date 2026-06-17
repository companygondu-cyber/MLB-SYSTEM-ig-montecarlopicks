#!/usr/bin/env python3
"""app.py — OMEGA MLB entry point. Thin CLI wrapper around pipeline functions."""
import argparse
import sys
from datetime import datetime, timedelta

from pipeline import (
    C,
    predict_live,
    backtest,
    sync_data,
    auto_sync_and_retrain,
    load_data,
    backup_db,
    restore_db,
)
from models.game import train_multiwindow
from features import build_all_features
import sync as savant_sync


def main():
    pa = argparse.ArgumentParser(description='OMEGA MLB v3')
    pa.add_argument('--mode', choices=['train', 'backtest', 'predict', 'sync', 'daily', 'full'], default='predict')
    pa.add_argument('--date', type=str, default=None)
    pa.add_argument('--beta', action='store_true')
    pa.add_argument('--guess-lineups', action='store_true', help='Infer unconfirmed lineups (BETA)')
    pa.add_argument('--skip-sync', action='store_true')
    pa.add_argument('--disable-filters', action='store_true', help='Disable streak/volatility filters')
    a = pa.parse_args()

    if a.mode in ('predict', 'sync'):
        backup_db()
        try:
            if a.mode == 'sync':
                if not a.skip_sync:
                    sync_data(30)
                    print(f"{C['C']}Sincronizando Statcast (Savant)...{C['X']}")
                    savant_sync.sync_savant_daily()
            else:
                predict_live(
                    a.date,
                    beta=a.beta,
                    guess_lineups=a.guess_lineups,
                    use_filters=not a.disable_filters,
                    skip_sync=a.skip_sync,
                )
        except Exception:
            import traceback
            traceback.print_exc()
            restore_db()
            sys.exit(1)
    elif a.mode in ('daily', 'full'):
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"\n{'='*70}")
        print(f"MLB DAILY UPDATE - {today_str}")
        print(f"{'='*70}")
        backup_db()
        try:
            print(f"\n[1/3] Syncing yesterday + Statcast...")
            sync_data(1)
            savant_sync.sync_savant_daily()
            from models.game import needs_retrain
            if needs_retrain(min_days=7):
                print(f"\n[2/3] Retraining models (weekly)...")
                df, pp = load_data()
                df, f, t = build_all_features(df, pp)
                train_multiwindow(df, f, t)
                print(f"   OK Models retrained ({len(df)} games, {len(f)} features)")
            else:
                print(f"\n[2/3] Retrain no necesario (modelos <7 días).")
            print(f"\n[3/3] Predicting today: {today_str}")
            predict_live(
                today_str,
                beta=a.beta,
                guess_lineups=a.guess_lineups,
                use_filters=not a.disable_filters,
                skip_sync=a.skip_sync,
            )
        except Exception:
            import traceback
            traceback.print_exc()
            restore_db()
            sys.exit(1)
        print(f"\n{'='*70}")
        print(f"MLB DAILY UPDATE COMPLETE")
        print(f"{'='*70}")
    elif a.mode == 'train':
        backup_db()
        df, pp = load_data()
        df, f, t = build_all_features(df, pp)
        train_multiwindow(df, f, t)
    elif a.mode == 'backtest':
        backtest(beta=a.beta)


if __name__ == '__main__':
    main()
