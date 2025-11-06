#!/usr/bin/env python3
"""Merge two SQLite databases for experiment tracking."""

import argparse
import sqlite3
import sys
from pathlib import Path


def merge_databases(source_db: str, target_db: str, dry_run: bool = False):
    """Merge source database into target database.
    
    Args:
        source_db: Path to source database (e.g., from cluster)
        target_db: Path to target database (e.g., local)
        dry_run: If True, only show what would be merged
    """
    source_path = Path(source_db)
    target_path = Path(target_db)
    
    if not source_path.exists():
        print(f"❌ Source database not found: {source_path}")
        sys.exit(1)
    
    if not target_path.exists():
        print(f"❌ Target database not found: {target_path}")
        sys.exit(1)
    
    # Connect to both databases
    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)
    
    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()
    
    # Get experiments from source
    source_cursor.execute("SELECT * FROM experiments")
    source_experiments = source_cursor.fetchall()
    
    # Get column names
    source_cursor.execute("PRAGMA table_info(experiments)")
    columns = [col[1] for col in source_cursor.fetchall()]
    
    print(f"\n{'=' * 80}")
    print(f"📊 Database Merge Summary")
    print(f"{'=' * 80}")
    print(f"Source DB: {source_db}")
    print(f"Target DB: {target_db}")
    print(f"Found {len(source_experiments)} experiments in source")
    
    # Check which experiments are new
    target_cursor.execute("SELECT run_name FROM experiments")
    existing_runs = {row[0] for row in target_cursor.fetchall()}
    
    new_experiments = [exp for exp in source_experiments if exp[1] not in existing_runs]  # exp[1] is run_name
    duplicate_experiments = [exp for exp in source_experiments if exp[1] in existing_runs]
    
    print(f"\n✅ New experiments to add: {len(new_experiments)}")
    print(f"⚠️  Duplicate experiments (will skip): {len(duplicate_experiments)}")
    
    if new_experiments:
        print("\nNew experiments:")
        for exp in new_experiments:
            run_name = exp[1]
            timestamp = exp[2]
            val_mse = exp[11] if len(exp) > 11 else None
            print(f"  - {run_name} (timestamp: {timestamp}, val_mse: {val_mse})")
    
    if duplicate_experiments:
        print("\nDuplicate experiments (skipping):")
        for exp in duplicate_experiments:
            run_name = exp[1]
            print(f"  - {run_name}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - No changes made")
        source_conn.close()
        target_conn.close()
        return
    
    # Insert new experiments
    if new_experiments:
        print(f"\n⚙️  Merging {len(new_experiments)} experiments...")
        
        placeholders = ','.join(['?' for _ in columns])
        insert_query = f"INSERT INTO experiments ({','.join(columns)}) VALUES ({placeholders})"
        
        for exp in new_experiments:
            try:
                target_cursor.execute(insert_query, exp)
                
                # Also merge training_history for this experiment
                run_name = exp[1]
                source_cursor.execute(
                    "SELECT * FROM training_history WHERE run_name = ?",
                    (run_name,)
                )
                history_rows = source_cursor.fetchall()
                
                if history_rows:
                    target_cursor.executemany(
                        "INSERT INTO training_history (id, run_name, epoch, train_loss, val_loss, train_mse, val_mse, epoch_time_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        history_rows
                    )
                    print(f"  ✓ Merged {run_name} with {len(history_rows)} epochs")
                else:
                    print(f"  ✓ Merged {run_name} (no training history)")
                    
            except sqlite3.IntegrityError as e:
                print(f"  ⚠️  Skipped {run_name}: {e}")
        
        target_conn.commit()
        print(f"\n✅ Merge complete! {len(new_experiments)} experiments added to {target_db}")
    else:
        print(f"\n✅ No new experiments to merge")
    
    # Optionally merge leaderboard snapshots
    source_cursor.execute("SELECT COUNT(*) FROM leaderboard_snapshots")
    source_snapshots = source_cursor.fetchone()[0]
    
    target_cursor.execute("SELECT COUNT(*) FROM leaderboard_snapshots")
    target_snapshots = target_cursor.fetchone()[0]
    
    if source_snapshots > 0:
        print(f"\n📊 Leaderboard snapshots: {source_snapshots} in source, {target_snapshots} in target")
        print(f"   (Leaderboard snapshots are NOT merged to avoid duplicates)")
    
    print(f"{'=' * 80}\n")
    
    source_conn.close()
    target_conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Merge experiment databases from cluster and local machines"
    )
    parser.add_argument(
        "source",
        help="Source database to merge FROM (e.g., runs_cluster.db)"
    )
    parser.add_argument(
        "target",
        help="Target database to merge INTO (e.g., experiments/runs.db)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without making changes"
    )
    
    args = parser.parse_args()
    
    merge_databases(args.source, args.target, args.dry_run)


if __name__ == "__main__":
    main()

