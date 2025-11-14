#!/usr/bin/env python3
"""Migrate SQLite data to PostgreSQL."""

import sys
import os
from pathlib import Path
from infrastructure.database import ExperimentDatabase

def migrate_sqlite_to_postgres(sqlite_path: str, database_url: str):
    """Migrate all data from SQLite to PostgreSQL.
    
    Args:
        sqlite_path: Path to SQLite database
        database_url: PostgreSQL connection string
    """
    
    # Connect to SQLite (force it)
    print(f"📂 Opening SQLite database: {sqlite_path}")
    os.environ.pop("DATABASE_URL", None)  # Temporarily remove to force SQLite
    sqlite_db = ExperimentDatabase(sqlite_path)
    
    # Get all data from SQLite
    experiments_df = sqlite_db.get_all_experiments()
    print(f"📊 Found {len(experiments_df)} experiments in SQLite")
    
    # Connect to Postgres
    print(f"\n🔗 Connecting to PostgreSQL...")
    os.environ["DATABASE_URL"] = database_url
    postgres_db = ExperimentDatabase()  # Will use DATABASE_URL
    
    if len(experiments_df) == 0:
        print("⚠️  No experiments to migrate")
        sqlite_db.close()
        postgres_db.close()
        return
    
    # Migrate experiments
    print(f"\n⚙️  Migrating {len(experiments_df)} experiments...")
    migrated = 0
    skipped = 0
    
    for idx, row in experiments_df.iterrows():
        run_name = row['run_name']
        try:
            # Check if already exists in Postgres
            existing = postgres_db.get_experiment(run_name)
            if existing:
                print(f"  ⏭️  Skipping {run_name} (already exists)")
                skipped += 1
                continue
            
            # Insert experiment
            import json
            config = json.loads(row['config_json'])
            exp_id = postgres_db.create_experiment(run_name, config)
            
            # Update all fields
            update_fields = {}
            for col in experiments_df.columns:
                if col not in ['id', 'run_name', 'timestamp', 'config_json']:
                    val = row[col]
                    if not (isinstance(val, float) and str(val) == 'nan'):  # Skip NaN values
                        update_fields[col] = val
            
            if update_fields:
                postgres_db.update_experiment(run_name, **update_fields)
            
            # Migrate training history
            history_df = sqlite_db.get_training_history(run_name)
            if len(history_df) > 0:
                for _, hist_row in history_df.iterrows():
                    postgres_db.add_training_epoch(
                        run_name=run_name,
                        epoch=int(hist_row['epoch']),
                        train_loss=float(hist_row['train_loss']) if hist_row['train_loss'] else None,
                        val_loss=float(hist_row['val_loss']) if hist_row['val_loss'] else None,
                        train_mse=float(hist_row['train_mse']) if hist_row.get('train_mse') else None,
                        val_mse=float(hist_row['val_mse']) if hist_row.get('val_mse') else None,
                        epoch_time=float(hist_row['epoch_time_seconds']) if hist_row.get('epoch_time_seconds') else None
                    )
                print(f"  ✅ Migrated {run_name} ({len(history_df)} epochs)")
            else:
                print(f"  ✅ Migrated {run_name} (no training history)")
            
            migrated += 1
            
        except Exception as e:
            print(f"  ❌ Failed to migrate {run_name}: {e}")
    
    print(f"\n{'=' * 80}")
    print(f"🎉 Migration complete!")
    print(f"   ✅ Migrated: {migrated} experiments")
    print(f"   ⏭️  Skipped: {skipped} experiments (already existed)")
    print(f"{'=' * 80}\n")
    
    # Optionally migrate leaderboard snapshots
    try:
        leaderboard_df = sqlite_db.get_leaderboard_history()
        if len(leaderboard_df) > 0:
            print(f"📊 Note: {len(leaderboard_df)} leaderboard snapshots NOT migrated")
            print(f"   (They can be re-scraped with: uv run python server_cli.py leaderboard)")
    except:
        pass
    
    sqlite_db.close()
    postgres_db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate SQLite database to PostgreSQL")
    parser.add_argument(
        "--sqlite",
        default="experiments/runs.db",
        help="Path to SQLite database (default: experiments/runs.db)"
    )
    parser.add_argument(
        "--postgres-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection string (default: from DATABASE_URL env var)"
    )
    
    args = parser.parse_args()
    
    if not args.postgres_url:
        print("❌ Error: PostgreSQL URL not provided")
        print("   Set DATABASE_URL environment variable or use --postgres-url")
        sys.exit(1)
    
    if not Path(args.sqlite).exists():
        print(f"❌ Error: SQLite database not found: {args.sqlite}")
        sys.exit(1)
    
    migrate_sqlite_to_postgres(args.sqlite, args.postgres_url)






