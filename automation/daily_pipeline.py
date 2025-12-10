#!/usr/bin/env python3
"""
Daily Pipeline Orchestrator.

Runs all stages in sequence with proper error handling, logging, and metrics.

Usage:
    python -m automation.daily_pipeline
    python -m automation.daily_pipeline --dry-run
    python -m automation.daily_pipeline --stage scrape
"""

import gzip
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from automation.config import PipelineConfig
from automation.stages import StageResult, StageStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Critical pipeline error that should halt execution."""
    pass


class DailyPipeline:
    """
    Orchestrates the daily scrape → enrich → train → report pipeline.

    Design principles:
    1. Each stage is idempotent and can be run independently
    2. Failures are logged but don't halt the pipeline (unless critical)
    3. All operations are logged with structured metrics
    4. Database backups are taken before destructive operations
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        run_id: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.config = config or PipelineConfig()
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dry_run = dry_run
        self.results: list[StageResult] = []
        self.start_time: Optional[datetime] = None
        self._setup_directories()

    def _setup_directories(self):
        """Ensure required directories exist."""
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        # Create run-specific log directory
        self.run_log_dir = self.config.log_dir / self.run_id
        self.run_log_dir.mkdir(exist_ok=True)

    def run(self, stages: Optional[list[str]] = None) -> bool:
        """
        Run the full pipeline or specific stages.

        Args:
            stages: Optional list of stage names to run. If None, runs all.

        Returns:
            True if pipeline completed successfully, False otherwise.
        """
        self.start_time = datetime.now()
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Starting daily pipeline run: {self.run_id}")

        # Define stage order and functions
        all_stages = [
            ('preflight', self._run_preflight),
            ('scrape', self._run_scrape),
            ('enrich', self._run_enrich),
            ('dedupe', self._run_dedupe),
            ('train', self._run_train),
            ('report', self._run_report),
            ('postflight', self._run_postflight),
        ]

        # Filter stages if specified
        if stages:
            all_stages = [(name, func) for name, func in all_stages if name in stages]
            logger.info(f"Running specific stages: {stages}")

        success = True
        for stage_name, stage_func in all_stages:
            stage_config = getattr(self.config, stage_name, None)

            # Skip disabled stages
            if stage_config and not stage_config.enabled:
                logger.info(f"Stage '{stage_name}' is disabled, skipping")
                continue

            try:
                result = self._run_stage(stage_name, stage_func)
                self.results.append(result)

                if result.status == StageStatus.FAILED:
                    if stage_config and not stage_config.continue_on_failure:
                        logger.error(f"Stage '{stage_name}' failed and continue_on_failure=False, halting pipeline")
                        success = False
                        break
                    else:
                        logger.warning(f"Stage '{stage_name}' failed but continuing pipeline")

            except PipelineError as e:
                logger.error(f"Critical pipeline error in '{stage_name}': {e}")
                success = False
                break

        # Generate summary
        self._generate_summary()
        return success

    def _run_stage(self, stage_name: str, stage_func: Callable) -> StageResult:
        """Run a single stage with timing and error handling."""
        logger.info(f"{'='*60}")
        logger.info(f"STAGE: {stage_name.upper()}")
        logger.info(f"{'='*60}")

        start = datetime.now()
        result = StageResult(
            stage_name=stage_name,
            status=StageStatus.RUNNING,
            started_at=start,
        )

        try:
            if self.dry_run and stage_name not in ['preflight', 'postflight']:
                logger.info(f"[DRY RUN] Would run stage: {stage_name}")
                result.status = StageStatus.SKIPPED
            else:
                result = stage_func(result)

        except Exception as e:
            logger.exception(f"Stage '{stage_name}' failed with exception")
            result.status = StageStatus.FAILED
            result.error_message = str(e)

        finally:
            result.finished_at = datetime.now()
            result.duration_seconds = (result.finished_at - start).total_seconds()
            logger.info(f"Stage '{stage_name}' completed: {result.status.value} ({result.duration_seconds:.1f}s)")

        return result

    # =========================================================================
    # PREFLIGHT STAGE
    # =========================================================================
    def _run_preflight(self, result: StageResult) -> StageResult:
        """
        Prepare for pipeline run:
        1. Kill stale processes
        2. Check disk space
        3. Validate database
        4. Backup database
        5. Mark old listings inactive
        """
        # 1. Kill stale processes
        killed = self._kill_stale_processes()
        result.metrics['stale_processes_killed'] = killed
        logger.info(f"Killed {killed} stale processes")

        # 2. Check disk space
        disk_free_gb = self._check_disk_space()
        result.metrics['disk_free_gb'] = disk_free_gb
        if disk_free_gb < self.config.min_disk_space_gb:
            raise PipelineError(f"Insufficient disk space: {disk_free_gb:.1f}GB < {self.config.min_disk_space_gb}GB")
        logger.info(f"Disk space: {disk_free_gb:.1f}GB free")

        # 3. Validate database
        if not self.config.db_path.exists():
            raise PipelineError(f"Database not found: {self.config.db_path}")
        if not self._validate_database():
            raise PipelineError("Database integrity check failed")
        logger.info("Database integrity: OK")

        # 4. Backup database
        backup_path = self._backup_database()
        result.metrics['backup_path'] = str(backup_path)
        logger.info(f"Database backed up to: {backup_path}")

        # 5. Mark old listings inactive
        marked = self._mark_inactive_listings()
        result.metrics['listings_marked_inactive'] = marked
        logger.info(f"Marked {marked} listings as inactive (not seen in {self.config.mark_inactive_days} days)")

        # 6. Get baseline stats
        stats = self._get_db_stats()
        result.metrics['baseline_stats'] = stats
        logger.info(f"Baseline: {stats['total']} total, {stats['active']} active, {stats['with_sqft']} with sqft")

        result.status = StageStatus.SUCCESS
        return result

    def _kill_stale_processes(self) -> int:
        """Kill stale scraper/enricher processes."""
        patterns = ['floorplan_enricher', 'ocr_enrich', 'scrapy crawl']
        killed = 0
        for pattern in patterns:
            try:
                subprocess.run(['pkill', '-f', pattern], capture_output=True)
                killed += 1
            except Exception:
                pass
        return killed

    def _check_disk_space(self) -> float:
        """Return free disk space in GB."""
        stat = os.statvfs(self.config.project_root)
        return (stat.f_bavail * stat.f_frsize) / (1024**3)

    def _validate_database(self) -> bool:
        """Run SQLite integrity check."""
        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            conn.close()
            return result == "ok"
        except Exception:
            return False

    def _backup_database(self) -> Path:
        """Create compressed backup of database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config.backup_dir / f"rentals_{timestamp}.db.gz"

        with open(self.config.db_path, 'rb') as f_in:
            with gzip.open(backup_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        return backup_path

    def _mark_inactive_listings(self) -> int:
        """Mark listings not seen recently as inactive."""
        cutoff = (datetime.utcnow() - timedelta(days=self.config.mark_inactive_days)).isoformat()
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE listings SET is_active = 0
            WHERE is_active = 1 AND last_seen < ?
        """, (cutoff,))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        return affected

    def _get_db_stats(self) -> dict:
        """Get current database statistics."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM listings")
        stats['total'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM listings WHERE is_active = 1")
        stats['active'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM listings WHERE size_sqft > 0")
        stats['with_sqft'] = cursor.fetchone()[0]
        conn.close()
        return stats

    # =========================================================================
    # SCRAPE STAGE
    # =========================================================================
    def _run_scrape(self, result: StageResult) -> StageResult:
        """Run all spiders with retry logic."""
        spider_results = {}
        total_items = 0
        failed_spiders = []

        for spider_config in self.config.spiders:
            spider_name = spider_config.name
            logger.info(f"Running spider: {spider_name}")

            success, items, error = self._run_spider_with_retry(spider_config)
            spider_results[spider_name] = {
                'success': success,
                'items': items,
                'error': error,
            }

            if success:
                total_items += items
                logger.info(f"  {spider_name}: OK ({items} items)")
            else:
                failed_spiders.append(spider_name)
                logger.warning(f"  {spider_name}: FAILED - {error}")

        result.items_processed = total_items
        result.items_failed = len(failed_spiders)
        result.metrics['spider_results'] = spider_results

        if failed_spiders:
            result.warnings.append(f"Failed spiders: {', '.join(failed_spiders)}")
            result.status = StageStatus.WARNING
        else:
            result.status = StageStatus.SUCCESS

        return result

    def _run_spider_with_retry(self, spider_config) -> tuple[bool, int, Optional[str]]:
        """Run a spider with exponential backoff retry."""
        for attempt, backoff in enumerate(spider_config.retry_backoff + [0], 1):
            logger.info(f"  Attempt {attempt}/{len(spider_config.retry_backoff) + 1}")

            success, items, error = self._run_single_spider(spider_config)
            if success:
                return True, items, None

            if backoff > 0:
                logger.info(f"  Retrying in {backoff}s...")
                time.sleep(backoff)

        return False, 0, error

    def _run_single_spider(self, spider_config) -> tuple[bool, int, Optional[str]]:
        """Run a single spider via subprocess."""
        settings_module = (
            "property_scraper.settings" if spider_config.requires_playwright
            else "property_scraper.settings_standard"
        )

        cmd = ["scrapy", "crawl", spider_config.name]
        env = os.environ.copy()
        env["SCRAPY_SETTINGS_MODULE"] = settings_module
        env["PYTHONPATH"] = str(self.config.project_root)

        log_file = self.run_log_dir / f"spider_{spider_config.name}.log"

        try:
            with open(log_file, "w") as f:
                proc = subprocess.run(
                    cmd,
                    cwd=self.config.project_root,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=spider_config.timeout_seconds,
                )

            if proc.returncode == 0:
                # Parse log to get item count
                items = self._parse_spider_log(log_file)
                return True, items, None
            else:
                return False, 0, f"Exit code {proc.returncode}"

        except subprocess.TimeoutExpired:
            return False, 0, f"Timeout after {spider_config.timeout_seconds}s"
        except Exception as e:
            return False, 0, str(e)

    def _parse_spider_log(self, log_file: Path) -> int:
        """Parse spider log to extract item count."""
        try:
            with open(log_file) as f:
                content = f.read()
            # Look for Scrapy stats line
            import re
            match = re.search(r"'item_scraped_count': (\d+)", content)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 0

    # =========================================================================
    # ENRICH STAGE
    # =========================================================================
    def _run_enrich(self, result: StageResult) -> StageResult:
        """Run enrichment for all sources."""
        enriched_total = 0
        enrichment_results = {}

        for source in self.config.enrich_sources:
            logger.info(f"Enriching: {source}")
            success, count = self._run_enricher(source)
            enrichment_results[source] = {'success': success, 'count': count}
            if success:
                enriched_total += count
                logger.info(f"  {source}: {count} enriched")
            else:
                logger.warning(f"  {source}: FAILED")

        result.items_processed = enriched_total
        result.metrics['enrichment_results'] = enrichment_results
        result.status = StageStatus.SUCCESS
        return result

    def _run_enricher(self, source: str) -> tuple[bool, int]:
        """Run floorplan enricher for a single source."""
        needs_playwright = source in ['savills', 'knightfrank', 'chestertons']
        settings_module = "property_scraper.settings" if needs_playwright else "property_scraper.settings_standard"

        cmd = [
            "scrapy", "crawl", "floorplan_enricher",
            "-a", f"source={source}",
            "-s", "HTTPCACHE_ENABLED=False",
        ]
        if self.config.enrich_limit_per_source:
            cmd.extend(["-a", f"limit={self.config.enrich_limit_per_source}"])

        env = os.environ.copy()
        env["SCRAPY_SETTINGS_MODULE"] = settings_module
        env["PYTHONPATH"] = str(self.config.project_root)

        log_file = self.run_log_dir / f"enrich_{source}.log"

        try:
            with open(log_file, "w") as f:
                proc = subprocess.run(
                    cmd,
                    cwd=self.config.project_root,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=self.config.enrich.timeout_seconds,
                )
            return proc.returncode == 0, self._parse_spider_log(log_file)
        except Exception:
            return False, 0

    # =========================================================================
    # DEDUPE STAGE
    # =========================================================================
    def _run_dedupe(self, result: StageResult) -> StageResult:
        """Run cross-source deduplication."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        # Copy sqft from agent sources to Rightmove records
        cursor.execute("""
            UPDATE listings
            SET size_sqft = (
                SELECT l2.size_sqft
                FROM listings l2
                WHERE l2.address_fingerprint = listings.address_fingerprint
                AND l2.size_sqft > 0
                AND l2.source IN ('savills', 'knightfrank', 'chestertons', 'foxtons')
                ORDER BY
                    CASE l2.source
                        WHEN 'savills' THEN 1
                        WHEN 'knightfrank' THEN 2
                        WHEN 'chestertons' THEN 3
                        WHEN 'foxtons' THEN 4
                    END
                LIMIT 1
            )
            WHERE source = 'rightmove'
            AND (size_sqft IS NULL OR size_sqft = 0)
            AND address_fingerprint IS NOT NULL
        """)
        merged = cursor.rowcount
        conn.commit()
        conn.close()

        result.items_processed = merged
        result.metrics['sqft_merged'] = merged
        logger.info(f"Merged sqft into {merged} Rightmove records")
        result.status = StageStatus.SUCCESS
        return result

    # =========================================================================
    # TRAIN STAGE
    # =========================================================================
    def _run_train(self, result: StageResult) -> StageResult:
        """Retrain model if sufficient new data."""
        # Check if we should retrain
        stats = self._get_db_stats()
        result.metrics['current_records'] = stats['with_sqft']

        # For now, always train (can add threshold logic later)
        logger.info(f"Training model with {stats['with_sqft']} records")

        cmd = ["python3", "rental_price_models_v14.py", "--quick"]
        log_file = self.run_log_dir / "train_model.log"

        try:
            with open(log_file, "w") as f:
                proc = subprocess.run(
                    cmd,
                    cwd=self.config.project_root,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=self.config.train.timeout_seconds,
                )

            if proc.returncode == 0:
                result.status = StageStatus.SUCCESS
                logger.info("Model training completed successfully")
            else:
                result.status = StageStatus.FAILED
                result.error_message = f"Training failed with exit code {proc.returncode}"

        except subprocess.TimeoutExpired:
            result.status = StageStatus.FAILED
            result.error_message = "Training timed out"

        return result

    # =========================================================================
    # REPORT STAGE
    # =========================================================================
    def _run_report(self, result: StageResult) -> StageResult:
        """Generate negotiation report."""
        cmd = ["python3", "scripts/generate_negotiation_report.py"]
        log_file = self.run_log_dir / "report.log"

        try:
            with open(log_file, "w") as f:
                proc = subprocess.run(
                    cmd,
                    cwd=self.config.project_root,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=self.config.report.timeout_seconds,
                )

            if proc.returncode == 0:
                result.status = StageStatus.SUCCESS
                logger.info("Report generated successfully")
            else:
                result.status = StageStatus.FAILED
                result.error_message = f"Report generation failed with exit code {proc.returncode}"

        except subprocess.TimeoutExpired:
            result.status = StageStatus.FAILED
            result.error_message = "Report generation timed out"

        return result

    # =========================================================================
    # POSTFLIGHT STAGE
    # =========================================================================
    def _run_postflight(self, result: StageResult) -> StageResult:
        """Cleanup and generate summary."""
        # Get final stats
        final_stats = self._get_db_stats()
        result.metrics['final_stats'] = final_stats

        # Compare with baseline
        baseline = self.results[0].metrics.get('baseline_stats', {}) if self.results else {}
        if baseline:
            result.metrics['new_listings'] = final_stats['total'] - baseline.get('total', 0)
            result.metrics['new_sqft'] = final_stats['with_sqft'] - baseline.get('with_sqft', 0)
            logger.info(f"New listings: {result.metrics['new_listings']}")
            logger.info(f"New sqft records: {result.metrics['new_sqft']}")

        # Cleanup old logs
        cleaned = self._cleanup_old_logs()
        result.metrics['logs_cleaned'] = cleaned
        logger.info(f"Cleaned {cleaned} old log files")

        # Cleanup old backups
        cleaned_backups = self._cleanup_old_backups()
        result.metrics['backups_cleaned'] = cleaned_backups
        logger.info(f"Cleaned {cleaned_backups} old backup files")

        result.status = StageStatus.SUCCESS
        return result

    def _cleanup_old_logs(self) -> int:
        """Remove logs older than keep_logs_days."""
        cutoff = datetime.now() - timedelta(days=self.config.keep_logs_days)
        cleaned = 0
        for log_file in self.config.log_dir.glob("*.log"):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff:
                log_file.unlink()
                cleaned += 1
        return cleaned

    def _cleanup_old_backups(self) -> int:
        """Remove backups older than keep_backups_days."""
        cutoff = datetime.now() - timedelta(days=self.config.keep_backups_days)
        cleaned = 0
        for backup_file in self.config.backup_dir.glob("*.db.gz"):
            if datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff:
                backup_file.unlink()
                cleaned += 1
        return cleaned

    # =========================================================================
    # SUMMARY
    # =========================================================================
    def _generate_summary(self):
        """Generate and log pipeline summary."""
        total_duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Total Duration: {total_duration / 60:.1f} minutes")
        logger.info("")

        for result in self.results:
            status_icon = {
                StageStatus.SUCCESS: "✓",
                StageStatus.WARNING: "⚠",
                StageStatus.FAILED: "✗",
                StageStatus.SKIPPED: "○",
            }.get(result.status, "?")
            logger.info(f"  {status_icon} {result.stage_name}: {result.status.value} ({result.duration_seconds:.0f}s)")
            if result.warnings:
                for warning in result.warnings:
                    logger.info(f"      ⚠ {warning}")

        # Save summary to file
        summary_file = self.run_log_dir / "summary.json"
        summary = {
            'run_id': self.run_id,
            'started_at': self.start_time.isoformat() if self.start_time else None,
            'finished_at': datetime.now().isoformat(),
            'total_duration_seconds': total_duration,
            'dry_run': self.dry_run,
            'stages': [r.to_dict() for r in self.results],
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSummary saved to: {summary_file}")


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run daily scrape pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--stage", action="append", help="Run specific stage(s) only")
    args = parser.parse_args()

    pipeline = DailyPipeline(dry_run=args.dry_run)
    success = pipeline.run(stages=args.stage)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
