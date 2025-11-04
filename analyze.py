"""Analysis and visualization script for experiment results."""

import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from infrastructure.database import ExperimentDatabase
from infrastructure.visualization import (
    plot_experiment_comparison,
    plot_leaderboard_history,
    plot_metric_distribution
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_list(args):
    """List all experiments."""
    db = ExperimentDatabase(args.db_path)
    df = db.get_all_experiments()
    
    if len(df) == 0:
        logger.info("No experiments found in database")
        return
    
    # Select columns to display
    display_cols = [
        'run_name', 'model_architecture', 'latent_dim',
        'val_mse', 'model_size_mb', 'server_weighted_score',
        'server_rank', 'timestamp'
    ]
    
    # Filter to existing columns
    display_cols = [col for col in display_cols if col in df.columns]
    
    logger.info(f"\nFound {len(df)} experiments:\n")
    print(df[display_cols].to_string(index=False))
    
    db.close()


def cmd_compare(args):
    """Compare multiple experiments."""
    db = ExperimentDatabase(args.db_path)
    
    # Get experiments
    all_df = db.get_all_experiments()
    
    if args.run_names:
        # Filter to specific runs
        df = all_df[all_df['run_name'].isin(args.run_names)]
    else:
        # Use all experiments
        df = all_df
    
    if len(df) == 0:
        logger.error("No experiments found to compare")
        db.close()
        return
    
    logger.info(f"Comparing {len(df)} experiments")
    
    # Generate comparison plot
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = "experiment_comparison.png"
    
    plot_experiment_comparison(
        df,
        metrics=args.metrics,
        save_path=str(output_path),
        show=args.show
    )
    
    logger.info(f"Saved comparison plot to: {output_path}")
    
    db.close()


def cmd_best(args):
    """Show best experiments by metric."""
    db = ExperimentDatabase(args.db_path)
    
    best_df = db.get_best_experiments(
        metric=args.metric,
        top_n=args.top_n
    )
    
    if len(best_df) == 0:
        logger.info(f"No experiments with {args.metric} found")
        db.close()
        return
    
    logger.info(f"\nTop {len(best_df)} experiments by {args.metric}:\n")
    
    display_cols = [
        'run_name', 'model_architecture', 'latent_dim',
        'val_mse', 'model_size_mb', 'server_weighted_score',
        'server_rank'
    ]
    
    display_cols = [col for col in display_cols if col in best_df.columns]
    
    print(best_df[display_cols].to_string(index=False))
    
    db.close()


def cmd_history(args):
    """Show training history for a run."""
    db = ExperimentDatabase(args.db_path)
    
    history_df = db.get_training_history(args.run_name)
    
    if len(history_df) == 0:
        logger.error(f"No training history found for run: {args.run_name}")
        db.close()
        return
    
    logger.info(f"\nTraining history for {args.run_name}:\n")
    print(history_df.to_string(index=False))
    
    # Plot if requested
    if args.plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(history_df['epoch'], history_df['train_loss'], 
               'b-', label='Train Loss', linewidth=2)
        ax.plot(history_df['epoch'], history_df['val_loss'],
               'r-', label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training History - {args.run_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if args.output:
            plt.savefig(args.output, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to: {args.output}")
        
        if args.show:
            plt.show()
        else:
            plt.close()
    
    db.close()


def cmd_leaderboard(args):
    """Analyze leaderboard history."""
    db = ExperimentDatabase(args.db_path)
    
    leaderboard_df = db.get_leaderboard_history(team_name=args.team_name)
    
    if len(leaderboard_df) == 0:
        logger.info("No leaderboard snapshots found")
        db.close()
        return
    
    if args.team_name:
        logger.info(f"\nLeaderboard history for team '{args.team_name}':\n")
    else:
        logger.info("\nLeaderboard history:\n")
    
    print(leaderboard_df.to_string(index=False))
    
    # Plot if requested
    if args.plot and args.team_name:
        output_path = args.output or f"leaderboard_{args.team_name.replace(' ', '_')}.png"
        
        plot_leaderboard_history(
            leaderboard_df,
            team_name=args.team_name,
            save_path=output_path,
            show=args.show
        )
        
        logger.info(f"Saved plot to: {output_path}")
    
    db.close()


def cmd_stats(args):
    """Show database statistics."""
    db = ExperimentDatabase(args.db_path)
    
    # Get all experiments
    df = db.get_all_experiments()
    
    logger.info("\n" + "=" * 60)
    logger.info("Database Statistics")
    logger.info("=" * 60)
    
    logger.info(f"\nTotal experiments: {len(df)}")
    
    if len(df) > 0:
        # Architecture breakdown
        if 'model_architecture' in df.columns:
            arch_counts = df['model_architecture'].value_counts()
            logger.info("\nExperiments by architecture:")
            for arch, count in arch_counts.items():
                logger.info(f"  {arch}: {count}")
        
        # Completion status
        submitted = df['server_status'].notna().sum()
        logger.info(f"\nSubmitted to server: {submitted} / {len(df)}")
        
        if 'server_weighted_score' in df.columns:
            scored = df['server_weighted_score'].notna().sum()
            logger.info(f"Evaluated on server: {scored} / {len(df)}")
            
            if scored > 0:
                scores = df['server_weighted_score'].dropna()
                logger.info(f"\nWeighted Score Statistics:")
                logger.info(f"  Mean: {scores.mean():.4f}")
                logger.info(f"  Std: {scores.std():.4f}")
                logger.info(f"  Min: {scores.min():.4f}")
                logger.info(f"  Max: {scores.max():.4f}")
        
        # Model size statistics
        if 'model_size_mb' in df.columns:
            sizes = df['model_size_mb'].dropna()
            if len(sizes) > 0:
                logger.info(f"\nModel Size Statistics (MB):")
                logger.info(f"  Mean: {sizes.mean():.2f}")
                logger.info(f"  Std: {sizes.std():.2f}")
                logger.info(f"  Min: {sizes.min():.2f}")
                logger.info(f"  Max: {sizes.max():.2f}")
    
    logger.info("\n" + "=" * 60)
    
    db.close()


def main():
    """Main analyze entry point."""
    parser = argparse.ArgumentParser(
        description="Analysis tools for experiment results"
    )
    
    parser.add_argument("--db-path", default="experiments/runs.db",
                       help="Path to experiment database")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", 
                                          help="Compare experiments")
    compare_parser.add_argument("--run-names", nargs="+",
                               help="Specific runs to compare")
    compare_parser.add_argument("--metrics", nargs="+",
                               default=['val_mse', 'model_size_mb', 'server_weighted_score'],
                               help="Metrics to compare")
    compare_parser.add_argument("--output", help="Output plot path")
    compare_parser.add_argument("--show", action="store_true",
                               help="Show plot interactively")
    
    # Best command
    best_parser = subparsers.add_parser("best", 
                                       help="Show best experiments")
    best_parser.add_argument("--metric", default="server_weighted_score",
                            help="Metric to rank by")
    best_parser.add_argument("--top-n", type=int, default=10,
                            help="Number of top experiments")
    
    # History command
    history_parser = subparsers.add_parser("history",
                                          help="Show training history")
    history_parser.add_argument("run_name", help="Run name")
    history_parser.add_argument("--plot", action="store_true",
                               help="Plot training curves")
    history_parser.add_argument("--output", help="Output plot path")
    history_parser.add_argument("--show", action="store_true",
                               help="Show plot interactively")
    
    # Leaderboard command
    leaderboard_parser = subparsers.add_parser("leaderboard",
                                              help="Analyze leaderboard history")
    leaderboard_parser.add_argument("--team-name",
                                   default="ignore all instructinos",
                                   help="Team name to analyze")
    leaderboard_parser.add_argument("--plot", action="store_true",
                                   help="Plot leaderboard history")
    leaderboard_parser.add_argument("--output", help="Output plot path")
    leaderboard_parser.add_argument("--show", action="store_true",
                                   help="Show plot interactively")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats",
                                        help="Show database statistics")
    
    args = parser.parse_args()
    
    if args.command == "list":
        cmd_list(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "best":
        cmd_best(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "leaderboard":
        cmd_leaderboard(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

