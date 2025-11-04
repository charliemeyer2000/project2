"""SQLite database for experiment tracking."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd


class ExperimentDatabase:
    """Manage experiment tracking in SQLite database."""
    
    def __init__(self, db_path: str = "experiments/runs.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                config_json TEXT NOT NULL,
                
                -- Model info
                model_architecture TEXT,
                latent_dim INTEGER,
                model_size_mb REAL,
                num_parameters INTEGER,
                
                -- Training metrics
                train_loss_final REAL,
                val_loss_final REAL,
                train_mse REAL,
                val_mse REAL,
                best_epoch INTEGER,
                total_epochs INTEGER,
                training_time_seconds REAL,
                
                -- Server metrics (NULL until submitted)
                server_submission_id INTEGER,
                server_status TEXT,
                server_weighted_score REAL,
                server_full_mse REAL,
                server_roi_mse REAL,
                server_latent_dim INTEGER,
                server_model_size_mb REAL,
                server_rank INTEGER,
                server_submitted_at TEXT,
                
                -- Paths
                checkpoint_path TEXT,
                torchscript_path TEXT,
                config_path TEXT,
                log_path TEXT,
                output_dir TEXT
            )
        """)
        
        # Leaderboard snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rank INTEGER NOT NULL,
                team TEXT NOT NULL,
                weighted_score REAL NOT NULL,
                latent_dim INTEGER NOT NULL,
                full_mse REAL NOT NULL,
                roi_mse REAL NOT NULL,
                model_size_mb REAL NOT NULL,
                submitted_at TEXT NOT NULL
            )
        """)
        
        # Training history table (for epoch-by-epoch tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                val_loss REAL,
                train_mse REAL,
                val_mse REAL,
                epoch_time_seconds REAL,
                FOREIGN KEY (run_name) REFERENCES experiments(run_name)
            )
        """)
        
        self.conn.commit()
    
    def create_experiment(self, run_name: str, config: Dict[str, Any]) -> int:
        """Create a new experiment entry.
        
        Args:
            run_name: Unique name for this run
            config: Full configuration dictionary
            
        Returns:
            Experiment ID
        """
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        config_json = json.dumps(config, indent=2)
        
        try:
            cursor.execute("""
                INSERT INTO experiments (run_name, timestamp, config_json)
                VALUES (?, ?, ?)
            """, (run_name, timestamp, config_json))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"Experiment '{run_name}' already exists!")
    
    def update_experiment(self, run_name: str, **kwargs):
        """Update experiment fields.
        
        Args:
            run_name: Experiment name
            **kwargs: Fields to update
        """
        if not kwargs:
            return
        
        # Build SET clause dynamically
        set_clause = ", ".join(f"{key} = ?" for key in kwargs.keys())
        values = list(kwargs.values()) + [run_name]
        
        cursor = self.conn.cursor()
        cursor.execute(f"""
            UPDATE experiments
            SET {set_clause}
            WHERE run_name = ?
        """, values)
        self.conn.commit()
    
    def add_training_epoch(self, run_name: str, epoch: int, 
                          train_loss: float, val_loss: float,
                          train_mse: Optional[float] = None,
                          val_mse: Optional[float] = None,
                          epoch_time: Optional[float] = None):
        """Add epoch-level training metrics.
        
        Args:
            run_name: Experiment name
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_mse: Training MSE (optional)
            val_mse: Validation MSE (optional)
            epoch_time: Time taken for epoch (optional)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO training_history 
            (run_name, epoch, train_loss, val_loss, train_mse, val_mse, epoch_time_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_name, epoch, train_loss, val_loss, train_mse, val_mse, epoch_time))
        self.conn.commit()
    
    def get_experiment(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Get experiment by name.
        
        Args:
            run_name: Experiment name
            
        Returns:
            Experiment dict or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE run_name = ?", (run_name,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_experiments(self) -> pd.DataFrame:
        """Get all experiments as DataFrame.
        
        Returns:
            DataFrame with all experiments
        """
        return pd.read_sql_query("SELECT * FROM experiments", self.conn)
    
    def get_training_history(self, run_name: str) -> pd.DataFrame:
        """Get training history for a run.
        
        Args:
            run_name: Experiment name
            
        Returns:
            DataFrame with epoch-by-epoch history
        """
        return pd.read_sql_query(
            "SELECT * FROM training_history WHERE run_name = ? ORDER BY epoch",
            self.conn,
            params=(run_name,)
        )
    
    def save_leaderboard_snapshot(self, leaderboard_data: List[Dict[str, Any]]):
        """Save leaderboard snapshot to database.
        
        Args:
            leaderboard_data: List of leaderboard entries
        """
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        
        for entry in leaderboard_data:
            cursor.execute("""
                INSERT INTO leaderboard_snapshots
                (timestamp, rank, team, weighted_score, latent_dim, 
                 full_mse, roi_mse, model_size_mb, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                entry['rank'],
                entry['team'],
                entry['weighted_score'],
                entry['latent_dim'],
                entry['full_mse'],
                entry['roi_mse'],
                entry['model_size_mb'],
                entry['submitted_at']
            ))
        
        self.conn.commit()
    
    def get_leaderboard_history(self, team_name: Optional[str] = None) -> pd.DataFrame:
        """Get leaderboard history over time.
        
        Args:
            team_name: Optional team name to filter by
            
        Returns:
            DataFrame with leaderboard history
        """
        if team_name:
            return pd.read_sql_query(
                "SELECT * FROM leaderboard_snapshots WHERE team = ? ORDER BY timestamp",
                self.conn,
                params=(team_name,)
            )
        return pd.read_sql_query(
            "SELECT * FROM leaderboard_snapshots ORDER BY timestamp, rank",
            self.conn
        )
    
    def get_best_experiments(self, metric: str = "server_weighted_score", 
                           top_n: int = 10) -> pd.DataFrame:
        """Get top N experiments by metric.
        
        Args:
            metric: Metric to sort by
            top_n: Number of top experiments to return
            
        Returns:
            DataFrame with top experiments
        """
        return pd.read_sql_query(f"""
            SELECT * FROM experiments 
            WHERE {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT ?
        """, self.conn, params=(top_n,))
    
    def generate_run_name(self, prefix: str = "") -> str:
        """Generate a unique run name.
        
        Args:
            prefix: Optional prefix for the run name
            
        Returns:
            Unique run name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            return f"{prefix}_{timestamp}"
        return f"run_{timestamp}"
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

