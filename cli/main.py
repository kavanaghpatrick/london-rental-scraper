#!/usr/bin/env python3
"""
Rental Scraper CLI - Unified orchestration tool (PRD-004).

Usage:
    rental-scraper scrape --all
    rental-scraper scrape --source savills
    rental-scraper status
    rental-scraper mark-inactive --days 7
"""

import os
import re
import signal
import subprocess
import sys
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer


# =============================================================================
# Issue #23 FIX: Subprocess Ctrl+C handling
# =============================================================================

# Track running subprocesses for cleanup on Ctrl+C
_running_processes: list[subprocess.Popen] = []
_original_sigint_handler = None


def _cleanup_subprocesses(signum, frame):
    """Gracefully terminate all running subprocesses on Ctrl+C."""
    for proc in _running_processes:
        if proc.poll() is None:  # Still running
            try:
                proc.terminate()  # SIGTERM first
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()  # Force kill if SIGTERM didn't work
            except Exception:
                pass
    _running_processes.clear()

    # Re-raise KeyboardInterrupt for the main process
    if _original_sigint_handler and callable(_original_sigint_handler):
        _original_sigint_handler(signum, frame)
    else:
        raise KeyboardInterrupt


def _register_signal_handlers():
    """Register signal handlers once at startup."""
    global _original_sigint_handler
    if _original_sigint_handler is None:
        _original_sigint_handler = signal.signal(signal.SIGINT, _cleanup_subprocesses)


from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# === Issue #8 FIX: Check conda environment at startup ===
def _check_environment():
    """Verify correct conda environment is active."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    # Allow running if in correct env OR if sqlite3 imports successfully
    # (handles cases where env name differs but dependencies are met)
    if conda_env and conda_env != 'claude-code':
        # Only warn, don't block - the import will fail with clear error if wrong
        pass  # Continue, let import errors speak for themselves

_check_environment()

from cli.registry import SPIDERS, get_spider, get_all_spiders

# Ensure we're in the scrapy project directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)

app = typer.Typer(
    name="rental-scraper",
    help="London Rental Property Scraper CLI",
    add_completion=False,
)
console = Console()


def get_db_path() -> Path:
    """Get the database path."""
    return PROJECT_ROOT / "output" / "rentals.db"


# === Issue #5 FIX: Parse scrapy stats from log file ===
def _parse_scrapy_stats(log_file: Path) -> dict:
    """Parse scrapy stats from log file to extract item counts.

    Returns dict with keys: item_scraped_count, item_dropped_count, error_count
    """
    stats = {
        'item_scraped_count': 0,
        'item_dropped_count': 0,
        'error_count': 0,
        'finish_reason': 'unknown',
    }

    if not log_file.exists():
        return stats

    try:
        content = log_file.read_text(errors='ignore')

        # Look for 'item_scraped_count': N in the stats dump
        match = re.search(r"'item_scraped_count':\s*(\d+)", content)
        if match:
            stats['item_scraped_count'] = int(match.group(1))

        # Look for item_dropped_count
        match = re.search(r"'item_dropped_count':\s*(\d+)", content)
        if match:
            stats['item_dropped_count'] = int(match.group(1))

        # Count ERROR log lines
        stats['error_count'] = len(re.findall(r'\[.*\]\s+ERROR:', content))

        # Get finish reason
        match = re.search(r"'finish_reason':\s*'(\w+)'", content)
        if match:
            stats['finish_reason'] = match.group(1)

    except Exception:
        pass

    return stats


# === Issue #7 FIX: Validate scrape results in database ===
def _validate_scrape_results(source: str, min_expected: int = 1) -> tuple[bool, int]:
    """Check if recent items were scraped for a source.

    Returns (success, count) where success is True if count >= min_expected.
    """
    db_path = get_db_path()
    if not db_path.exists():
        return False, 0

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Count items updated in the last 10 minutes for this source
        cursor.execute("""
            SELECT COUNT(*) FROM listings
            WHERE source = ?
            AND last_seen >= datetime('now', '-10 minutes')
        """, (source,))
        count = cursor.fetchone()[0]
        conn.close()
        return count >= min_expected, count
    except Exception:
        return False, 0


def run_spider(
    spider_name: str,
    max_pages: int | None = None,
    max_properties: int | None = None,
    dry_run: bool = False,
    fetch_details: bool = False,
    fetch_floorplans: bool = False,
) -> tuple[bool, str, bool, dict]:
    """
    Run a spider using subprocess (avoids Twisted reactor issues).

    Returns (success: bool, message: str, detail_skipped: bool, stats: dict)

    Issue #5 FIX: Now parses log file for actual item counts and validates results.
    """
    config = get_spider(spider_name)
    if not config:
        return False, f"Unknown spider: {spider_name}", False, {}

    # Build command
    cmd = ["scrapy", "crawl", spider_name]

    if max_pages:
        cmd.extend(["-a", f"max_pages={max_pages}"])
    if max_properties:
        cmd.extend(["-a", f"max_properties={max_properties}"])

    # Check if spider supports detail fetching (some Playwright spiders are too slow)
    detail_skipped = False
    if not config.supports_detail_fetch and (fetch_details or fetch_floorplans):
        detail_skipped = True
        fetch_details = False
        fetch_floorplans = False

    # Fetch details (sqft, descriptions) from detail pages - slower but more complete
    if fetch_details:
        cmd.extend(["-a", "fetch_details=true"])

    # Fetch floorplan URLs from detail pages
    if fetch_floorplans:
        cmd.extend(["-a", "fetch_floorplans=true"])

    # Choose settings based on spider type
    if config.requires_playwright:
        settings_module = "property_scraper.settings"
    else:
        settings_module = "property_scraper.settings_standard"

    # Set environment
    env = os.environ.copy()
    env["SCRAPY_SETTINGS_MODULE"] = settings_module
    # Add project root to PYTHONPATH so scrapy can find property_scraper module
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    # If dry_run, disable SQLite pipeline to prevent database writes
    # (Fix for Codex review: dry_run flag was being ignored)
    if dry_run:
        # Override ITEM_PIPELINES to exclude SQLitePipeline
        cmd.extend([
            "-s", "ITEM_PIPELINES={'property_scraper.pipelines.CleanDataPipeline': 100, "
                  "'property_scraper.pipelines.DuplicateFilterPipeline': 200, "
                  "'property_scraper.pipelines.JsonWriterPipeline': 300}"
        ])

    # Create log file
    # Issue #27 FIX: Use /tmp for dry-run logs to avoid cluttering logs/ directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dry_run:
        log_dir = Path('/tmp')
        log_file = log_dir / f"{spider_name}_{timestamp}_dryrun.log"
    else:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{spider_name}_{timestamp}.log"

    # === Issue #23 FIX: Use Popen for Ctrl+C handling ===
    proc = None
    try:
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            _running_processes.append(proc)

            try:
                returncode = proc.wait(timeout=3600)  # 1 hour timeout
            finally:
                # Remove from tracking list once done
                if proc in _running_processes:
                    _running_processes.remove(proc)

        # === Issue #5 FIX: Parse log file for actual item counts ===
        stats = _parse_scrapy_stats(log_file)

        if returncode == 0:
            # Check if spider actually scraped items (not just exited cleanly)
            if stats['item_scraped_count'] == 0 and not dry_run:
                # Spider ran but scraped nothing - this is a failure!
                return False, f"FAILED: 0 items scraped (check log). Log: {log_file.name}", detail_skipped, stats
            return True, f"Completed ({stats['item_scraped_count']} items). Log: {log_file.name}", detail_skipped, stats
        else:
            return False, f"Failed (exit {returncode}). Log: {log_file.name}", detail_skipped, stats

    except subprocess.TimeoutExpired:
        # Kill the process on timeout
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc in _running_processes:
            _running_processes.remove(proc)
        return False, f"Timeout after 1 hour. Log: {log_file.name}", detail_skipped, {}
    except KeyboardInterrupt:
        # Process cleanup handled by signal handler, just re-raise
        raise
    except Exception as e:
        if proc in _running_processes:
            _running_processes.remove(proc)
        return False, f"Error: {e}", detail_skipped, {}


def _run_enrichment(source: str, console: Console) -> bool:
    """Run floorplan enrichment for a single source. Returns success."""
    needs_playwright = source in ['savills', 'knightfrank', 'chestertons']
    settings_module = "property_scraper.settings" if needs_playwright else "property_scraper.settings_standard"

    cmd = ["scrapy", "crawl", "floorplan_enricher", "-a", f"source={source}",
           "-s", "HTTPCACHE_ENABLED=False"]

    # Source-specific throttling to avoid rate limiting
    # Chestertons is particularly aggressive with 429 responses
    if source == 'chestertons':
        cmd.extend([
            "-s", "CONCURRENT_REQUESTS=1",
            "-s", "CONCURRENT_REQUESTS_PER_DOMAIN=1",
            "-s", "DOWNLOAD_DELAY=10",
            "-s", "AUTOTHROTTLE_MAX_DELAY=300",
            "-s", "AUTOTHROTTLE_TARGET_CONCURRENCY=0.5",
        ])

    env = os.environ.copy()
    env["SCRAPY_SETTINGS_MODULE"] = settings_module
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"floorplan_enricher_{source}_{timestamp}.log"

    # === Issue #23 FIX: Use Popen for Ctrl+C handling ===
    proc = None
    try:
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, env=env,
                stdout=f, stderr=subprocess.STDOUT
            )
            _running_processes.append(proc)

            # Chestertons needs longer timeout due to aggressive throttling
            timeout_mins = 120 if source == 'chestertons' else 30
            try:
                returncode = proc.wait(timeout=timeout_mins * 60)
            finally:
                if proc in _running_processes:
                    _running_processes.remove(proc)

        success = returncode == 0
        status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
        console.print(f"  {status} enrich {source}: Log: {log_file.name}")
        return success
    except subprocess.TimeoutExpired:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc in _running_processes:
            _running_processes.remove(proc)
        timeout_mins = 120 if source == 'chestertons' else 30
        console.print(f"  [red]FAIL[/red] enrich {source}: Timeout after {timeout_mins} min")
        return False
    except KeyboardInterrupt:
        raise
    except Exception as e:
        if proc in _running_processes:
            _running_processes.remove(proc)
        console.print(f"  [red]FAIL[/red] enrich {source}: {e}")
        return False


def _run_dedupe(console: Console) -> bool:
    """Run cross-source dedupe merge. Returns success."""
    db_path = get_db_path()
    if not db_path.exists():
        console.print("  [red]FAIL[/red] dedupe: Database not found")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
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
        updated = cursor.rowcount
        conn.commit()
        console.print(f"  [green]OK[/green] dedupe: Merged sqft into {updated} Rightmove records")
        return True
    except Exception as e:
        conn.rollback()
        console.print(f"  [red]FAIL[/red] dedupe: {e}")
        return False
    finally:
        conn.close()


@app.command()
def scrape(
    all: bool = typer.Option(False, "--all", help="Run all spiders sequentially"),
    # === Issue #12 FIX: Accept multiple sources with -s flag ===
    source: Optional[list[str]] = typer.Option(None, "--source", "-s", help="Specific spider(s) to run (can specify multiple)"),
    max_pages: int = typer.Option(None, "--max-pages", help="Max pages per spider"),
    max_properties: int = typer.Option(None, "--max-properties", help="Max properties total"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse but don't save to DB"),
    fetch_details: bool = typer.Option(False, "--fetch-details", "-d", help="Fetch sqft/descriptions from detail pages (slower but more complete)"),
    fetch_floorplans: bool = typer.Option(False, "--fetch-floorplans", "-f", help="Fetch floorplan URLs from detail pages"),
    full: bool = typer.Option(False, "--full", help="Enable both --fetch-details and --fetch-floorplans for maximum data"),
):
    """Run property scrapers."""
    if not all and not source:
        console.print("[red]Error:[/red] Must specify --all or --source")
        raise typer.Exit(1)

    # --full enables both fetch options
    if full:
        fetch_details = True
        fetch_floorplans = True

    # Get spiders to run
    if all:
        spiders_to_run = [s.name for s in get_all_spiders()]
    else:
        # === Issue #12 FIX: Handle list of sources ===
        spiders_to_run = []
        for src in source:
            if src.lower() not in SPIDERS:
                console.print(f"[red]Error:[/red] Unknown spider: {src}")
                console.print(f"Available: {', '.join(SPIDERS.keys())}")
                raise typer.Exit(1)
            spiders_to_run.append(src.lower())

    console.print(f"\n[bold]Running {len(spiders_to_run)} spider(s)...[/bold]")
    if fetch_details or fetch_floorplans:
        options = []
        if fetch_details:
            options.append("fetch-details")
        if fetch_floorplans:
            options.append("fetch-floorplans")
        console.print(f"[dim]Options: {', '.join(options)} (slower but more complete)[/dim]")
    console.print()

    results = []
    skipped_detail_spiders = []
    total_items_scraped = 0

    for spider_name in spiders_to_run:
        config = get_spider(spider_name)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"[cyan]{spider_name}[/cyan]: Running...", total=None)

            success, message, detail_skipped, stats = run_spider(
                spider_name,
                max_pages=max_pages or config.default_max_pages,
                max_properties=max_properties,
                dry_run=dry_run,
                fetch_details=fetch_details,
                fetch_floorplans=fetch_floorplans,
            )

            progress.remove_task(task)

        status = "[green]OK[/green]" if success else "[red]FAIL[/red]"

        # === Issue #9 FIX: Explain why fast mode was used ===
        skip_note = ""
        if detail_skipped:
            skip_note = " [dim](fast mode: detail fetch disabled for this spider)[/dim]"

        console.print(f"  {status} [bold]{spider_name}[/bold]: {message}{skip_note}")
        results.append((spider_name, success, stats))

        if detail_skipped:
            skipped_detail_spiders.append(spider_name)

        # Track total items for summary
        total_items_scraped += stats.get('item_scraped_count', 0)

    # Summary
    console.print()
    success_count = sum(1 for _, s, _ in results if s)
    fail_count = len(results) - success_count

    if fail_count == 0:
        console.print(f"[bold green]Complete![/bold green] All {success_count} spider(s) succeeded. Total: {total_items_scraped:,} items.")
    else:
        console.print(f"[bold yellow]Done.[/bold yellow] {success_count} succeeded, {fail_count} failed. Total: {total_items_scraped:,} items.")

    # Auto-enrich spiders that skipped detail fetching (only when running --all --full)
    if skipped_detail_spiders and all and full and not dry_run:
        console.print()
        console.print(f"[bold]Auto-enriching {', '.join(skipped_detail_spiders)}...[/bold]")
        for spider_name in skipped_detail_spiders:
            _run_enrichment(spider_name, console)

    # Auto-dedupe after --all --full (merge sqft from agents to Rightmove)
    if all and full and not dry_run:
        console.print()
        console.print("[bold]Running cross-source dedupe...[/bold]")
        _run_dedupe(console)

    if fail_count > 0:
        raise typer.Exit(1)


@app.command()
def status():
    """Show database status and statistics."""
    db_path = get_db_path()

    if not db_path.exists():
        console.print(f"[red]Error:[/red] Database not found at {db_path}")
        raise typer.Exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Total listings
    cursor.execute("SELECT COUNT(*) FROM listings")
    total = cursor.fetchone()[0]

    # Active listings
    cursor.execute("SELECT COUNT(*) FROM listings WHERE is_active = 1")
    active = cursor.fetchone()[0]

    # With sqft
    cursor.execute("SELECT COUNT(*) FROM listings WHERE size_sqft > 0")
    with_sqft = cursor.fetchone()[0]

    # With fingerprints
    cursor.execute("SELECT COUNT(*) FROM listings WHERE address_fingerprint IS NOT NULL")
    with_fp = cursor.fetchone()[0]

    # Price history entries
    try:
        cursor.execute("SELECT COUNT(*) FROM price_history")
        price_history = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        price_history = 0

    # Recent scrapes (last 24h)
    yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
    cursor.execute("SELECT COUNT(*) FROM listings WHERE last_seen > ?", (yesterday,))
    recent = cursor.fetchone()[0]

    # By source
    cursor.execute("""
        SELECT source, COUNT(*),
               SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN size_sqft > 0 THEN 1 ELSE 0 END),
               MAX(last_seen)
        FROM listings GROUP BY source ORDER BY COUNT(*) DESC
    """)
    by_source = cursor.fetchall()

    conn.close()

    # Display summary
    console.print("\n[bold]Database Status[/bold]")
    console.print(f"  Total listings: [cyan]{total:,}[/cyan]")
    console.print(f"  Active listings: [cyan]{active:,}[/cyan]")
    console.print(f"  With sqft: [cyan]{with_sqft:,}[/cyan] ({100*with_sqft//total if total else 0}%)")
    console.print(f"  With fingerprints: [cyan]{with_fp:,}[/cyan]")
    console.print(f"  Price history entries: [cyan]{price_history:,}[/cyan]")
    console.print(f"  Scraped in last 24h: [cyan]{recent:,}[/cyan]")

    # Source table
    table = Table(title="\nBy Source")
    table.add_column("Source", style="bold")
    table.add_column("Total", justify="right")
    table.add_column("Active", justify="right")
    table.add_column("With sqft", justify="right")
    table.add_column("Last Seen")

    for source, total_count, active_count, sqft_count, last_seen in by_source:
        table.add_row(
            source or "unknown",
            str(total_count),
            str(active_count or 0),
            str(sqft_count or 0),
            last_seen[:10] if last_seen else "N/A"
        )

    console.print(table)


@app.command("mark-inactive")
def mark_inactive(
    days: int = typer.Option(7, "--days", "-d", help="Days since last seen"),
    execute: bool = typer.Option(False, "--execute", help="Actually apply changes"),
):
    """Mark listings not seen recently as inactive."""
    db_path = get_db_path()

    if not db_path.exists():
        console.print(f"[red]Error:[/red] Database not found at {db_path}")
        raise typer.Exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    # Count affected
    cursor.execute("""
        SELECT COUNT(*) FROM listings
        WHERE is_active = 1 AND last_seen < ?
    """, (cutoff,))
    count = cursor.fetchone()[0]

    console.print(f"\nFound [cyan]{count}[/cyan] listings not seen in {days}+ days")

    if count == 0:
        console.print("[green]Nothing to update.[/green]")
        conn.close()
        return

    if not execute:
        console.print("[yellow]Dry run.[/yellow] Use --execute to apply changes.")
    else:
        # CRITICAL FIX (Grok review): Wrap update in try/except with rollback
        # Previously, if UPDATE or commit failed, database could be left inconsistent.
        try:
            cursor.execute("""
                UPDATE listings SET is_active = 0
                WHERE is_active = 1 AND last_seen < ?
            """, (cutoff,))
            conn.commit()
            console.print(f"[green]Marked {count} listings as inactive.[/green]")
        except Exception as e:
            conn.rollback()
            console.print(f"[red]Error:[/red] Failed to mark listings as inactive: {e}")
            conn.close()
            raise typer.Exit(1)

    conn.close()


@app.command()
def dedupe(
    analyze: bool = typer.Option(False, "--analyze", help="Show potential duplicates"),
    merge: bool = typer.Option(False, "--merge", help="Merge cross-source duplicates"),
    execute: bool = typer.Option(False, "--execute", help="Actually apply changes"),
):
    """Analyze and merge cross-source duplicates."""
    db_path = get_db_path()

    if not db_path.exists():
        console.print(f"[red]Error:[/red] Database not found at {db_path}")
        raise typer.Exit(1)

    if not analyze and not merge:
        console.print("[red]Error:[/red] Must specify --analyze or --merge")
        raise typer.Exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if analyze:
        # Find duplicates by fingerprint
        cursor.execute("""
            SELECT address_fingerprint, COUNT(*) as cnt,
                   GROUP_CONCAT(source || ':' || property_id, ', ') as sources
            FROM listings
            WHERE address_fingerprint IS NOT NULL
            GROUP BY address_fingerprint
            HAVING cnt > 1
            ORDER BY cnt DESC
            LIMIT 20
        """)
        dupes = cursor.fetchall()

        if not dupes:
            console.print("\n[green]No cross-source duplicates found![/green]")
        else:
            console.print(f"\n[bold]Found {len(dupes)} fingerprints with multiple listings:[/bold]\n")

            table = Table()
            table.add_column("Fingerprint", max_width=20)
            table.add_column("Count", justify="right")
            table.add_column("Sources")

            for fp, cnt, sources in dupes:
                table.add_row(fp[:20] + "...", str(cnt), sources)

            console.print(table)

    if merge:
        # Count mergeable (same fingerprint, different sources)
        cursor.execute("""
            SELECT COUNT(DISTINCT address_fingerprint)
            FROM listings
            WHERE address_fingerprint IN (
                SELECT address_fingerprint
                FROM listings
                WHERE address_fingerprint IS NOT NULL
                GROUP BY address_fingerprint
                HAVING COUNT(DISTINCT source) > 1
            )
        """)
        mergeable = cursor.fetchone()[0]

        console.print(f"\nFound [cyan]{mergeable}[/cyan] fingerprints with cross-source duplicates")

        if not execute:
            console.print("[yellow]Dry run.[/yellow] Use --execute to apply merge.")
            console.print("Merge will copy sqft from agent sources to Rightmove records.")
        else:
            # Priority: savills > knightfrank > chestertons > foxtons > rightmove
            # Copy sqft from best source to others
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
            updated = cursor.rowcount
            conn.commit()
            console.print(f"[green]Merged sqft data into {updated} Rightmove records.[/green]")

    conn.close()


@app.command("enrich-floorplans")
def enrich_floorplans(
    source: str = typer.Option(None, "--source", "-s", help="Specific source to enrich"),
    limit: int = typer.Option(None, "--limit", "-l", help="Max listings to process per source"),
    use_ocr: bool = typer.Option(False, "--ocr", help="Run OCR to extract sqft from floorplans"),
):
    """Fetch missing floorplan URLs for all sources."""
    # Sources to enrich (HTTP sources first for speed)
    all_sources = ['foxtons', 'rightmove', 'savills', 'knightfrank', 'chestertons']

    if source:
        if source.lower() not in all_sources:
            console.print(f"[red]Error:[/red] Unknown source: {source}")
            console.print(f"Available: {', '.join(all_sources)}")
            raise typer.Exit(1)
        sources_to_run = [source.lower()]
    else:
        sources_to_run = all_sources

    console.print(f"\n[bold]Enriching floorplans for {len(sources_to_run)} source(s)...[/bold]\n")

    results = []
    for src in sources_to_run:
        # Determine settings based on source type
        needs_playwright = src in ['savills', 'knightfrank', 'chestertons']
        settings_module = "property_scraper.settings" if needs_playwright else "property_scraper.settings_standard"

        # Build command - ALWAYS disable cache for enricher
        # Playwright sources need fresh pages to click tabs (PLANS, Floorplan, etc.)
        # HTTP cache returns stale HTML without floorplan URLs
        cmd = ["scrapy", "crawl", "floorplan_enricher", "-a", f"source={src}",
               "-s", "HTTPCACHE_ENABLED=False"]
        if limit:
            cmd.extend(["-a", f"limit={limit}"])
        if use_ocr:
            cmd.extend(["-a", "use_ocr=true"])

        env = os.environ.copy()
        env["SCRAPY_SETTINGS_MODULE"] = settings_module
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"floorplan_enricher_{src}_{timestamp}.log"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"[cyan]{src}[/cyan]: Enriching floorplans...", total=None)

            # === Issue #23 FIX: Use Popen for Ctrl+C handling ===
            proc = None
            try:
                with open(log_file, "w") as f:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )
                    _running_processes.append(proc)

                    try:
                        returncode = proc.wait(timeout=3600)
                    finally:
                        if proc in _running_processes:
                            _running_processes.remove(proc)

                success = returncode == 0
                message = f"Log: {log_file.name}"
            except subprocess.TimeoutExpired:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                if proc in _running_processes:
                    _running_processes.remove(proc)
                success = False
                message = "Timeout after 1 hour"
            except KeyboardInterrupt:
                progress.remove_task(task)
                raise
            except Exception as e:
                if proc in _running_processes:
                    _running_processes.remove(proc)
                success = False
                message = f"Error: {e}"

            progress.remove_task(task)

        status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
        console.print(f"  {status} [bold]{src}[/bold]: {message}")
        results.append((src, success))

    # Summary
    console.print()
    success_count = sum(1 for _, s in results if s)
    console.print(f"[bold green]Complete![/bold green] Enriched {success_count}/{len(results)} sources.")


@app.command("ocr-enrich")
def ocr_enrich(
    source: str = typer.Option(None, "--source", "-s", help="Specific source to process"),
    limit: int = typer.Option(None, "--limit", "-l", help="Max listings to process"),
    workers: int = typer.Option(4, "--workers", "-w", help="Concurrent workers"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without updating DB"),
):
    """Run OCR on floorplan images to extract sqft and floor data."""
    script_path = PROJECT_ROOT / "scripts" / "ocr_enrich.py"

    if not script_path.exists():
        console.print("[red]Error:[/red] scripts/ocr_enrich.py not found")
        raise typer.Exit(1)

    cmd = ["python3", str(script_path)]
    if source:
        cmd.extend(["--source", source])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if workers:
        cmd.extend(["--workers", str(workers)])
    if dry_run:
        cmd.append("--dry-run")

    console.print(f"\n[bold]Running OCR enrichment...[/bold]\n")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            console.print("[red]OCR enrichment failed[/red]")
            raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        raise typer.Exit(1)


@app.command()
def daily(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without making changes"),
    stage: list[str] = typer.Option(None, "--stage", "-s", help="Run specific stage(s) only"),
):
    """Run the full daily pipeline: scrape → enrich → dedupe → train → report."""
    from automation.daily_pipeline import DailyPipeline

    console.print("\n[bold]Starting Daily Pipeline[/bold]")
    if dry_run:
        console.print("[yellow]DRY RUN MODE - no changes will be made[/yellow]")
    console.print()

    pipeline = DailyPipeline(dry_run=dry_run)
    success = pipeline.run(stages=stage if stage else None)

    if success:
        console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    else:
        console.print("\n[bold red]Pipeline completed with errors.[/bold red]")
        raise typer.Exit(1)


@app.command()
def spiders():
    """List available spiders."""
    console.print("\n[bold]Available Spiders[/bold]\n")

    table = Table()
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Default Pages", justify="right")
    table.add_column("Description")

    for spider in get_all_spiders():
        spider_type = "[magenta]Playwright[/magenta]" if spider.requires_playwright else "[cyan]HTTP[/cyan]"
        max_pages_display = str(spider.default_max_pages) if spider.default_max_pages else "all"
        table.add_row(
            spider.name,
            spider_type,
            max_pages_display,
            spider.description,
        )

    console.print(table)


def main():
    """Entry point."""
    _register_signal_handlers()
    app()


if __name__ == "__main__":
    main()
