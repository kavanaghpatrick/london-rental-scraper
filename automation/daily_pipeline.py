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
import pickle
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

        # Define stage order and functions.
        # Core loop (user ask: "model retrains after every scrape, everything served"):
        #   scrape -> enrich -> dedupe -> RETRAIN(canonical) -> EXPORT artifacts
        #   -> SYNC to Postgres/Neon -> deploy (trigger).
        all_stages = [
            ('preflight', self._run_preflight),
            ('scrape', self._run_scrape),
            ('enrich', self._run_enrich),
            ('dedupe', self._run_dedupe),
            ('train', self._run_train),      # retrain canonical (v20) -> rental_model_canonical.pkl
            ('export', self._run_export),    # model.json/features.json/similar_listings.json/predictions.json
            ('sync', self._run_sync),        # mirror canonical SQLite -> Neon Postgres (dry-run in pipeline)
            ('deploy', self._run_deploy),    # trigger dashboard/extension refresh (gated)
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

        # 5. Mark old listings inactive (cycle-relative; never in dry-run)
        marked = self._mark_inactive_listings(dry_run=self.dry_run)
        result.metrics['listings_marked_inactive'] = marked
        if self.dry_run:
            logger.info(f"[DRY RUN] Would mark {marked} listings inactive (no write performed)")
        else:
            logger.info(f"Marked {marked} listings as inactive (cycle-relative, {self.config.mark_inactive_days}d window)")

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
        """Create compressed backup of database, then VERIFY it's readable + valid.

        A corrupt backup is worse than no backup (false sense of safety). After
        gzipping, we re-open the .gz, write it to a temp file, and run a SQLite
        integrity_check — raising if the backup can't be read or is corrupt (dataeng
        #38 GAP A). This catches a bad backup at CREATION, not at restore time.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config.backup_dir / f"rentals_{timestamp}.db.gz"

        with open(self.config.db_path, 'rb') as f_in:
            with gzip.open(backup_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Verify: decompress to a temp file and integrity-check it.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with gzip.open(backup_path, 'rb') as f_in:
                with open(tmp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            conn = sqlite3.connect(tmp_path)
            try:
                ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                conn.close()
            if ok != "ok":
                backup_path.unlink(missing_ok=True)
                raise PipelineError(f"Backup verification FAILED ({ok}); removed corrupt backup {backup_path}")
        finally:
            tmp_path.unlink(missing_ok=True)

        return backup_path

    def _mark_inactive_listings(self, dry_run: bool = False) -> int:
        """Mark listings not seen in the last N days inactive — CYCLE-RELATIVE.

        BUG FIX (#25): the previous version used wall-clock `utcnow() - N days`. On a
        FROZEN/STALE snapshot (e.g. the canonical DB whose last scrape was months ago)
        every row is older than `now - N days`, so this zeroed ALL is_active — which
        re-empties the dashboard (the original prod-empty bug). Worse, it ran even in
        --dry-run because preflight is exempt from the dry-skip.

        Fixes:
          * Cutoff is relative to the DATA's own clock: max(last_seen) - N days. On a
            fresh scrape max(last_seen) ≈ now, so behavior is unchanged; on a stale
            snapshot the recent rows of THAT cycle stay active.
          * Frozen-snapshot guard: if the newest row is itself older than N days by
            wall clock, the data isn't from a live cycle — skip entirely rather than
            zero everything.
          * Honors dry_run: counts what WOULD change, writes nothing.
        """
        conn = sqlite3.connect(self.config.db_path)
        try:
            cursor = conn.cursor()
            row = cursor.execute("SELECT MAX(last_seen) FROM listings").fetchone()
            max_last_seen = row[0] if row else None
            if not max_last_seen:
                return 0

            # Frozen-snapshot guard: if even the newest listing is older than the
            # window by wall clock, treat the DB as a frozen snapshot and do nothing.
            wall_cutoff = (datetime.utcnow() - timedelta(days=self.config.mark_inactive_days)).isoformat()
            if max_last_seen < wall_cutoff:
                logger.warning(
                    f"Frozen-snapshot guard: newest last_seen={max_last_seen[:10]} is older than "
                    f"{self.config.mark_inactive_days}d — SKIPPING mark-inactive to avoid zeroing a stale DB"
                )
                return 0

            # Cycle-relative cutoff: relative to the data's own latest timestamp.
            try:
                anchor = datetime.fromisoformat(max_last_seen)
            except ValueError:
                anchor = datetime.utcnow()
            cutoff = (anchor - timedelta(days=self.config.mark_inactive_days)).isoformat()

            count = cursor.execute(
                "SELECT COUNT(*) FROM listings WHERE is_active = 1 AND last_seen < ?",
                (cutoff,),
            ).fetchone()[0]

            # Defense-in-depth (dataeng's request): even with the cycle-relative cutoff,
            # ABORT if this single pass would flip more than half of the currently-active
            # listings inactive. A correct daily cycle retires a small tail, never the
            # bulk — a >50% flip means the date logic or the data is wrong, so refuse to
            # write rather than risk re-emptying the dashboard/model.
            active_now = cursor.execute(
                "SELECT COUNT(*) FROM listings WHERE is_active = 1"
            ).fetchone()[0]
            if active_now > 0 and count > 0.5 * active_now:
                logger.error(
                    f"mark-inactive ABORTED: would flip {count}/{active_now} "
                    f"({100*count/active_now:.0f}%) of active listings inactive — exceeds 50% "
                    f"safety threshold. Refusing to write (likely bad cutoff or stale data)."
                )
                return 0

            if dry_run:
                return count  # no write

            cursor.execute(
                "UPDATE listings SET is_active = 0 WHERE is_active = 1 AND last_seen < ?",
                (cutoff,),
            )
            affected = cursor.rowcount
            conn.commit()
            return affected
        finally:
            conn.close()

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

    def _run_subprocess(self, cmd: list[str], log_name: str) -> bool:
        """Run a child command, tee output to a per-run log, return success.

        Shared helper for the train/export/sync/deploy stages. Honors the relevant
        stage timeout via the calling stage's config where applicable; uses the
        total_timeout as a safety ceiling otherwise. Logs the command for auditability.
        """
        log_file = self.run_log_dir / log_name
        logger.info(f"$ {' '.join(cmd)}  (log: {log_name})")
        try:
            with open(log_file, "w") as f:
                proc = subprocess.run(
                    cmd,
                    cwd=self.config.project_root,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=self.config.total_timeout_seconds,
                )
            if proc.returncode != 0:
                logger.warning(f"Command exited {proc.returncode}: {' '.join(cmd)} (see {log_name})")
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return False
        except Exception as e:
            logger.error(f"Command errored: {' '.join(cmd)}: {e}")
            return False

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
    # TRAIN STAGE (canonical retrain)
    # =========================================================================
    def _run_train(self, result: StageResult) -> StageResult:
        """Retrain the CANONICAL model (v20) after the scrape.

        Calls retrain_canonical.py (chosen by the modeler bake-off, MODEL_DECISION.md)
        which trains on the recency-independent set from a copy of the canonical DB
        and writes deterministic artifacts:
            output/rental_model_canonical.pkl
            output/rental_model_canonical_features.pkl
            output/rental_model_canonical_meta.json

        Idempotent: re-running overwrites the same deterministic paths. We verify the
        artifacts exist afterward and smoke-test the CV metrics against a floor so a
        regressed retrain is flagged (but does not, by itself, halt the pipeline).
        """
        stats = self._get_db_stats()
        result.metrics['current_records'] = stats['with_sqft']
        logger.info(f"Retraining canonical model (db has {stats['with_sqft']} sqft records)")

        # retrain_canonical.py reads a --db; in this local pipeline the canonical
        # SQLite is the source of truth. (CI is a fresh checkout, so reading the file
        # directly is safe per the modeler's note; locally there is no concurrent writer
        # by this point because the scrape stage has already finished.)
        cmd = [
            "python3", self.config.retrain_script,
            "--db", str(self.config.db_path),
        ]
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
        except subprocess.TimeoutExpired:
            result.status = StageStatus.FAILED
            result.error_message = "Canonical retrain timed out"
            return result

        if proc.returncode != 0:
            result.status = StageStatus.FAILED
            result.error_message = f"Retrain failed with exit code {proc.returncode} (see {log_file.name})"
            return result

        # Verify the deterministic artifacts were produced.
        missing = [
            str(p) for p in (
                self.config.canonical_model_path,
                self.config.canonical_features_path,
                self.config.canonical_meta_path,
            ) if not p.exists()
        ]
        if missing:
            result.status = StageStatus.FAILED
            result.error_message = f"Retrain finished but artifacts missing: {missing}"
            return result

        # Verify the inference-stats sidecar was produced. retrain_canonical.py now
        # emits output/rental_model_canonical_inference.json IN THE SAME RUN (modeler
        # folded gen_inference_stats in), so we don't run a separate gen step — we just
        # confirm it landed. canonical_predict.build_features(inference=True) loads it to
        # fix single-row postcode_freq/area_freq degeneration; a fresh pkl with a stale/
        # missing inference.json silently regresses single-row predict to the ~£3,430 bug.
        inference_json = self.config.project_root / "output" / "rental_model_canonical_inference.json"
        if inference_json.exists():
            result.metrics['inference_stats_present'] = True
            logger.info("rental_model_canonical_inference.json present (single-row freq maps)")
        else:
            result.warnings.append(
                "inference.json MISSING after retrain — single-row predict will regress to the degenerate "
                "fallback. retrain_canonical.py should emit it; check the train log."
            )

        # Smoke-test the CV metrics against the floor (MODEL_DECISION.md).
        try:
            with open(self.config.canonical_meta_path) as f:
                meta = json.load(f)
            cv = meta.get('cv_metrics_5fold', {})
            r2 = cv.get('R2')
            mae = cv.get('MAE')
            result.metrics['retrain_r2'] = r2
            result.metrics['retrain_mae'] = mae
            result.metrics['retrain_n_features'] = meta.get('n_features')
            result.metrics['retrain_n_samples'] = meta.get('n_samples')
            logger.info(f"Canonical retrain: R2={r2}, MAE=£{mae}, n_features={meta.get('n_features')}")

            regressed = False
            if r2 is not None and r2 < self.config.retrain_min_r2:
                regressed = True
                result.warnings.append(f"R2 {r2:.4f} below floor {self.config.retrain_min_r2}")
            if mae is not None and mae > self.config.retrain_max_mae:
                regressed = True
                result.warnings.append(f"MAE £{mae:,.0f} above ceiling £{self.config.retrain_max_mae:,.0f}")

            result.status = StageStatus.WARNING if regressed else StageStatus.SUCCESS
            if regressed:
                logger.warning("Canonical retrain REGRESSED below the smoke-test floor")
            else:
                logger.info("Canonical retrain passed smoke-test floor")
        except Exception as e:
            # Artifacts exist but meta unreadable: succeed with a warning rather than fail.
            result.warnings.append(f"Could not verify retrain metrics: {e}")
            result.status = StageStatus.WARNING

        return result

    # =========================================================================
    # EXPORT STAGE (orchestrate artifact generators owned by #7)
    # =========================================================================
    def _run_export(self, result: StageResult) -> StageResult:
        """Export served artifacts from the freshly-retrained canonical model.

        OWNERSHIP: the GENERATORS are owned by artifacts (#7). This stage only
        ORCHESTRATES them in sequence and verifies output, per the lead's ruling.
            1. chrome model.json + features.json  (from canonical pkl via xgboost save_model)
            2. similar_listings.json              (scripts/export_similar_listings.py)
            3. predictions.json                   (cache; best-effort)
        """
        exported = {}
        model_json = self.config.project_root / "chrome-extension" / "api" / "model.json"
        features_json = self.config.project_root / "chrome-extension" / "api" / "features.json"

        # --- 1. Chrome model.json + features.json from the CANONICAL model ---
        # Call the modeler's blessed entrypoint canonical_predict.export_to_chrome()
        # (lead's ruling: import, don't reimplement). It loads the canonical model +
        # the exact training feature order and writes both as ONE matched pair via
        # the Booster JSON (fixing the earlier xgboost wrapper save_model quirk that
        # exited non-zero). We still verify the served pair against the canonical
        # features pkl as a belt-and-braces check.
        chrome_script = (
            "import canonical_predict as cp\n"
            "cp.export_to_chrome('chrome-extension/api')\n"
        )
        exit_ok = self._run_subprocess(
            ["python3", "-c", chrome_script], "export_chrome.log"
        )

        # Truth = does the served pair match the canonical model? (order-sensitive)
        count_ok = False
        model_ok = model_json.exists() and model_json.stat().st_size > 1024
        try:
            canon_feats = pickle.loads(self.config.canonical_features_path.read_bytes())
            served_feats = json.loads(features_json.read_text()) if features_json.exists() else []
            result.metrics['served_feature_count'] = len(served_feats)
            result.metrics['canonical_feature_count'] = len(canon_feats)
            count_ok = list(served_feats) == list(canon_feats)
            if not count_ok:
                result.warnings.append(
                    f"served features ({len(served_feats)}) != canonical ({len(canon_feats)}) or order differs"
                )
        except Exception as e:
            result.warnings.append(f"feature verification failed: {e}")

        ok_chrome = count_ok and model_ok and exit_ok
        exported['chrome_model'] = ok_chrome
        if ok_chrome:
            logger.info(
                f"Chrome artifacts verified: features.json ({result.metrics.get('served_feature_count')}) "
                f"matches canonical order; model.json {model_json.stat().st_size // 1024} KB"
            )
        else:
            result.warnings.append("Chrome model/features export not verified (export error, missing model.json, or feature mismatch)")

        # --- 2. similar_listings.json ---
        ok_similar = self._run_subprocess(
            ["python3", self.config.similar_listings_script], "export_similar.log"
        )
        exported['similar_listings'] = ok_similar
        similar_json = self.config.project_root / "chrome-extension" / "api" / "similar_listings.json"
        if ok_similar and similar_json.exists():
            size_kb = similar_json.stat().st_size / 1024
            # Detect an EMPTY export: export_similar_listings.py filters is_active=1.
            # On a frozen snapshot where everything aged to is_active=0, this yields {}.
            try:
                n_similar = len(json.loads(similar_json.read_text()))
            except Exception:
                n_similar = -1
            result.metrics['similar_listings_count'] = n_similar
            logger.info(f"similar_listings.json exported ({size_kb:.1f} KB, {n_similar} listings)")
            if n_similar == 0:
                result.warnings.append(
                    "similar_listings.json is EMPTY: 0 active listings (is_active=1) in the DB. "
                    "On a fresh scrape this fills; on a frozen/aged snapshot it stays empty — "
                    "served comps will be empty. Data-state issue (not a code bug)."
                )
        else:
            result.warnings.append("similar_listings.json export incomplete")

        result.metrics['exports'] = exported
        # Stage is SUCCESS if the served model pair made it; warnings note partial.
        if exported.get('chrome_model'):
            result.status = StageStatus.WARNING if result.warnings else StageStatus.SUCCESS
        else:
            result.status = StageStatus.FAILED
            result.error_message = "Canonical model export failed (served model not refreshed)"
        return result

    # =========================================================================
    # SYNC STAGE (mirror canonical SQLite -> Neon Postgres)
    # =========================================================================
    def _run_sync(self, result: StageResult) -> StageResult:
        """Mirror the canonical SQLite DB into Neon Postgres.

        OWNERSHIP: serving (#8) owns scripts/sync_sqlite_to_postgres.py. This stage
        CALLS it. The pipeline ALWAYS runs it in DRY-RUN (no --execute): the real
        prod load is lead-gated behind --execute --i-have-rotated-the-secret and is
        run manually after secret rotation. So an automated pipeline run never writes
        prod, but it DOES validate schema parity and report the row-count delta.
        """
        if not os.environ.get("POSTGRES_URL"):
            result.warnings.append("POSTGRES_URL not set — skipping prod-sync dry-run")
            result.status = StageStatus.SKIPPED
            logger.info("POSTGRES_URL unset; sync stage skipped (local-only run)")
            return result

        # Dry-run only: NEVER pass --execute from the automated pipeline.
        ok = self._run_subprocess(
            ["python3", self.config.sync_script, "--sqlite", str(self.config.db_path)],
            "sync_postgres.log",
        )
        result.metrics['sync_dry_run'] = ok
        if ok:
            logger.info("Prod-sync DRY-RUN completed (no writes). Real load is lead-gated.")
            result.status = StageStatus.SUCCESS
        else:
            result.warnings.append("Prod-sync dry-run reported an error (see sync_postgres.log)")
            result.status = StageStatus.WARNING
        return result

    # =========================================================================
    # DEPLOY STAGE (trigger only; gated)
    # =========================================================================
    def _run_deploy(self, result: StageResult) -> StageResult:
        """Trigger the downstream deploy (dashboard/extension refresh).

        GATED: live deploy / prod writes are disabled by default (config.deploy_enabled
        = False) until the lead confirms (secret rotation). When disabled this stage is
        a logged no-op that records what it WOULD trigger, so the loop is complete and
        auditable without performing an outward-facing action.
        """
        if not self.config.deploy_enabled:
            logger.info("Deploy disabled (lead-gated). Would trigger: dashboard/extension artifact refresh.")
            result.metrics['deploy'] = 'skipped (gated)'
            result.status = StageStatus.SKIPPED
            return result

        # When enabled, the deploy is performed by committing refreshed artifacts in CI
        # (generate-predictions.yml / daily-scrape.yml). Locally there is nothing to push.
        logger.info("Deploy enabled: artifact commit/push is handled by CI workflows.")
        result.metrics['deploy'] = 'delegated-to-ci'
        result.status = StageStatus.SUCCESS
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

    def _cleanup_old_backups(self, keep_min: int = 5) -> int:
        """Remove backups older than keep_backups_days — but ALWAYS keep the newest
        `keep_min` regardless of age (dataeng #38).

        The old version deleted EVERY *.db.gz older than keep_backups_days with no
        floor. On a frozen snapshot or any pipeline gap of >keep_backups_days, the
        next run's cleanup would purge ALL backups → zero copies. The keep_min floor
        guarantees we never drop below N recent backups even if they're all "old".
        """
        backups = sorted(
            self.config.backup_dir.glob("*.db.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        cutoff = datetime.now() - timedelta(days=self.config.keep_backups_days)
        cleaned = 0
        for backup_file in backups[keep_min:]:  # never touch the newest keep_min
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
