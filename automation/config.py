"""
Pipeline configuration with sensible defaults.

All timeouts, retry counts, and thresholds are centralized here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SpiderConfig:
    """Configuration for a single spider."""
    name: str
    requires_playwright: bool
    timeout_seconds: int = 3600  # 1 hour default
    retry_count: int = 3
    retry_backoff: list[int] = field(default_factory=lambda: [5, 15, 45])


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    enabled: bool = True
    timeout_seconds: int = 3600
    continue_on_failure: bool = True  # If False, halt pipeline on stage failure


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    db_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "output" / "rentals.db")
    backup_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "output" / "backups")
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    # Global settings
    total_timeout_seconds: int = 7200  # 2 hours max
    mark_inactive_days: int = 7
    min_disk_space_gb: float = 1.0

    # Spiders (order matters: HTTP first for speed, then Playwright)
    spiders: list[SpiderConfig] = field(default_factory=lambda: [
        # HTTP spiders (fast)
        SpiderConfig(name="rightmove", requires_playwright=False, timeout_seconds=1800),
        SpiderConfig(name="foxtons", requires_playwright=False, timeout_seconds=1200),
        # Playwright spiders (slower, browser-based)
        SpiderConfig(name="savills", requires_playwright=True, timeout_seconds=2400),
        SpiderConfig(name="knightfrank", requires_playwright=True, timeout_seconds=1800),
        SpiderConfig(name="chestertons", requires_playwright=True, timeout_seconds=1800),
    ])

    # Stage configurations
    preflight: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=300,
        continue_on_failure=False,  # Must pass preflight
    ))
    scrape: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=5400,  # 90 min for all spiders
        continue_on_failure=True,  # Continue even if some spiders fail
    ))
    enrich: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=3600,
        continue_on_failure=True,
    ))
    dedupe: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=600,
        continue_on_failure=True,
    ))
    train: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=1800,  # Canonical retrain on full data; allow more than --quick
        continue_on_failure=True,
    ))
    export: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=900,
        continue_on_failure=True,  # A failed export shouldn't strand a good retrain
    ))
    sync: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=900,
        continue_on_failure=True,
    ))
    deploy: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=300,
        continue_on_failure=True,
    ))
    report: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=300,
        continue_on_failure=True,
    ))
    postflight: StageConfig = field(default_factory=lambda: StageConfig(
        timeout_seconds=300,
        continue_on_failure=True,
    ))

    # Training thresholds
    min_new_records_for_retrain: int = 50  # Minimum new records to trigger retraining
    retrain_percentage_threshold: float = 0.02  # Or 2% growth since last training

    # Canonical model retrain (task #9 / #6). retrain_canonical.py reads --db and
    # writes output/rental_model_canonical{,_features}.pkl + _meta.json.
    retrain_script: str = "retrain_canonical.py"
    canonical_model_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "output" / "rental_model_canonical.pkl")
    canonical_features_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "output" / "rental_model_canonical_features.pkl")
    canonical_meta_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "output" / "rental_model_canonical_meta.json")
    # Smoke-test floor (modeler MODEL_DECISION.md): alert if a retrain regresses below these.
    retrain_min_r2: float = 0.78
    retrain_max_mae: float = 1800.0

    # Export targets (artifacts owned by #7; the pipeline ORCHESTRATES their generators)
    similar_listings_script: str = "scripts/export_similar_listings.py"

    # Prod-sync (serving #8 owns scripts/sync_sqlite_to_postgres.py; pipeline CALLS it).
    # The pipeline NEVER passes --execute; real prod load is lead-gated (secret rotation).
    sync_script: str = "scripts/sync_sqlite_to_postgres.py"
    # Deploy is a trigger only; disabled until lead confirms (no live prod writes from cron).
    deploy_enabled: bool = False

    # Validation thresholds
    min_listings_per_spider: int = 10  # Warn if spider returns fewer
    max_price_change_percentage: float = 50.0  # Flag suspicious price changes

    # Enrichment settings
    enrich_sources: list[str] = field(default_factory=lambda: [
        'foxtons', 'rightmove', 'savills', 'knightfrank', 'chestertons'
    ])
    enrich_limit_per_source: Optional[int] = None  # None = no limit

    # Cleanup settings
    keep_logs_days: int = 30
    keep_backups_days: int = 7

    def get_spider(self, name: str) -> Optional[SpiderConfig]:
        """Get spider config by name."""
        for spider in self.spiders:
            if spider.name == name:
                return spider
        return None

    def http_spiders(self) -> list[SpiderConfig]:
        """Get HTTP-only spiders."""
        return [s for s in self.spiders if not s.requires_playwright]

    def playwright_spiders(self) -> list[SpiderConfig]:
        """Get Playwright spiders."""
        return [s for s in self.spiders if s.requires_playwright]
