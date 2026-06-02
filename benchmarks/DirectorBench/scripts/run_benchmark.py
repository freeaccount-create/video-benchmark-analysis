"""
CLI entry point for DirectorBench.

Commands:
  evaluate   Run evaluation on a single video
  batch      Run evaluation on a directory of metadata files
"""

from __future__ import annotations

import click

from directorbench.metadata.loader import VideoLoader
from directorbench.personalization.user_profile import UserProfile
from directorbench.pipeline import BenchmarkPipeline


@click.group()
def cli():
    """DirectorBench - Benchmark framework for minute-long AI-generated videos."""
    pass


@cli.command()
@click.option("--metadata", required=True, help="Path to video metadata JSON file.")
@click.option("--config", default="config/default_config.yaml", help="Path to config YAML.")
@click.option("--user-profile", default=None, help="Path to user profile YAML.")
@click.option("--output-dir", default="./reports", help="Directory to save reports.")
def evaluate(metadata: str, config: str, user_profile: str | None, output_dir: str):
    """Evaluate a single video from its metadata file."""
    # TODO: Load UserProfile if provided
    # TODO: Instantiate BenchmarkPipeline
    # TODO: Load VideoMetadata from metadata path
    # TODO: Call pipeline.run(video_metadata)
    # TODO: Call pipeline.save_report(report, output_dir)
    # TODO: Print summary to console
    raise NotImplementedError


@cli.command()
@click.option("--dataset-dir", required=True, help="Directory containing metadata JSON files.")
@click.option("--config", default="config/default_config.yaml", help="Path to config YAML.")
@click.option("--user-profile", default=None, help="Path to user profile YAML.")
@click.option("--output-dir", default="./reports", help="Directory to save reports.")
def batch(dataset_dir: str, config: str, user_profile: str | None, output_dir: str):
    """Evaluate all videos in a dataset directory."""
    # TODO: Load BenchmarkDataset from dataset_dir
    # TODO: Instantiate BenchmarkPipeline
    # TODO: Call pipeline.run_batch(dataset)
    # TODO: Print summary table of all results
    raise NotImplementedError


if __name__ == "__main__":
    cli()
