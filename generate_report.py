"""Generate experiment tracking report from database."""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from infrastructure.database import ExperimentDatabase
from infrastructure.server import ServerAPI

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_markdown_report(db_path: str, output_path: str, 
                            team_name: str = "ignore all instructinos"):
    """Generate a markdown report from the database.
    
    Args:
        db_path: Path to database
        output_path: Path to save markdown report
        team_name: Team name for leaderboard lookup
    """
    db = ExperimentDatabase(db_path)
    
    # Get all experiments
    experiments_df = db.get_all_experiments()
    
    # Get leaderboard history
    leaderboard_df = db.get_leaderboard_history(team_name=team_name)
    
    # Start building report
    report = []
    report.append("# 📊 Experiment Tracking Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Team:** {team_name}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Summary statistics
    report.append("## 📈 Summary Statistics")
    report.append("")
    report.append(f"**Total Experiments:** {len(experiments_df)}")
    
    if len(experiments_df) > 0:
        submitted = experiments_df['server_status'].notna().sum()
        report.append(f"**Submitted to Server:** {submitted}")
        
        if 'server_weighted_score' in experiments_df.columns:
            scored = experiments_df['server_weighted_score'].notna().sum()
            report.append(f"**Evaluated by Server:** {scored}")
            
            if scored > 0:
                best_score = experiments_df['server_weighted_score'].max()
                best_run = experiments_df.loc[
                    experiments_df['server_weighted_score'].idxmax(), 
                    'run_name'
                ]
                report.append(f"**Best Weighted Score:** {best_score:.4f} ({best_run})")
                
                if 'server_rank' in experiments_df.columns:
                    best_rank = experiments_df['server_rank'].min()
                    if not pd.isna(best_rank):
                        report.append(f"**Best Rank:** #{int(best_rank)}")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Submitted experiments table
    if len(experiments_df) > 0:
        submitted_df = experiments_df[experiments_df['server_status'].notna()].copy()
        
        if len(submitted_df) > 0:
            report.append("## 🏆 Submitted Models")
            report.append("")
            report.append("| # | Run Name | Latent Dim | Model Size (MB) | Val MSE | Server Full MSE | Server ROI MSE | Weighted Score | Rank | Status |")
            report.append("|---|----------|------------|-----------------|---------|-----------------|----------------|----------------|------|--------|")
            
            for idx, row in submitted_df.iterrows():
                rank_str = f"#{int(row['server_rank'])}" if pd.notna(row.get('server_rank')) else "-"
                score_str = f"{row['server_weighted_score']:.4f}" if pd.notna(row.get('server_weighted_score')) else "-"
                full_mse_str = f"{row['server_full_mse']:.6f}" if pd.notna(row.get('server_full_mse')) else "-"
                roi_mse_str = f"{row['server_roi_mse']:.6f}" if pd.notna(row.get('server_roi_mse')) else "-"
                val_mse_str = f"{row['val_mse']:.6f}" if pd.notna(row.get('val_mse')) else "-"
                model_size_str = f"{row['model_size_mb']:.2f}" if pd.notna(row.get('model_size_mb')) else "-"
                
                report.append(
                    f"| {idx+1} | {row['run_name']} | {row['latent_dim']} | "
                    f"{model_size_str} | {val_mse_str} | {full_mse_str} | "
                    f"{roi_mse_str} | {score_str} | {rank_str} | {row['server_status']} |"
                )
            
            report.append("")
    
    report.append("---")
    report.append("")
    
    # All experiments table
    if len(experiments_df) > 0:
        report.append("## 🔬 All Experiments")
        report.append("")
        report.append("| # | Run Name | Architecture | Latent Dim | Epochs | Val MSE | Model Size (MB) | Parameters | Status |")
        report.append("|---|----------|--------------|------------|--------|---------|-----------------|------------|--------|")
        
        for idx, row in experiments_df.iterrows():
            val_mse_str = f"{row['val_mse']:.6f}" if pd.notna(row.get('val_mse')) else "-"
            model_size_str = f"{row['model_size_mb']:.2f}" if pd.notna(row.get('model_size_mb')) else "-"
            params_str = f"{int(row['num_parameters']):,}" if pd.notna(row.get('num_parameters')) else "-"
            epochs_str = f"{int(row['total_epochs'])}" if pd.notna(row.get('total_epochs')) else "-"
            status = row.get('server_status', 'Not submitted')
            
            report.append(
                f"| {idx+1} | {row['run_name']} | {row.get('model_architecture', '-')} | "
                f"{row.get('latent_dim', '-')} | {epochs_str} | {val_mse_str} | "
                f"{model_size_str} | {params_str} | {status} |"
            )
        
        report.append("")
    
    report.append("---")
    report.append("")
    
    # Leaderboard history
    if len(leaderboard_df) > 0:
        report.append("## 📊 Leaderboard History")
        report.append("")
        report.append(f"Showing history for team: **{team_name}**")
        report.append("")
        report.append("| Timestamp | Rank | Weighted Score | Latent Dim | Full MSE | ROI MSE | Model Size (MB) |")
        report.append("|-----------|------|----------------|------------|----------|---------|-----------------|")
        
        for _, row in leaderboard_df.iterrows():
            report.append(
                f"| {row['timestamp']} | #{row['rank']} | {row['weighted_score']:.4f} | "
                f"{row['latent_dim']} | {row['full_mse']:.6f} | "
                f"{row['roi_mse']:.6f} | {row['model_size_mb']:.2f} |"
            )
        
        report.append("")
    
    report.append("---")
    report.append("")
    report.append("*This report is auto-generated from the experiment database.*")
    report.append("")
    report.append("To regenerate: `python generate_report.py`")
    report.append("")
    
    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✅ Generated report: {output_path}")
    
    db.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate experiment tracking report from database"
    )
    parser.add_argument("--db-path", default="experiments/runs.db",
                       help="Path to database")
    parser.add_argument("--output", default="EXPERIMENTS_REPORT.md",
                       help="Output markdown file")
    parser.add_argument("--team-name", default="ignore all instructinos",
                       help="Team name for leaderboard lookup")
    
    args = parser.parse_args()
    
    generate_markdown_report(args.db_path, args.output, args.team_name)


if __name__ == "__main__":
    import pandas as pd
    main()

