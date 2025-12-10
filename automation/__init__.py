"""
Automation package for daily scrape pipeline.

Usage:
    python -m automation.daily_pipeline
    python -m cli.main daily
"""

from automation.daily_pipeline import DailyPipeline
from automation.config import PipelineConfig

__all__ = ['DailyPipeline', 'PipelineConfig']
