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
import subprocess
import sys
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

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


def run_spider(
    spider_name: str,
    max_pages: int | None = None,
    max_properties: int | None = None,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """
    Run a spider using subprocess (avoids Twisted reactor issues).

    Returns (success: bool, message: str)
    """
    config = get_spider(spider_name)
    if not config:
        return False, f"Unknown spider: {spider_name}"

    # Build command
    cmd = ["scrapy", "crawl", spider_name]

    if max_pages:
        cmd.extend(["-a", f"max_pages={max_pages}"])
    if max_properties:
        cmd.extend(["-a", f"max_properties={max_properties}"])

    # Choose settings based on spider type
    if config.requires_playwright:
        settings_module = "property_scraper.settings"
    else:
        settings_module = "property_scraper.settings_standard"

    # Set environment
    env = os.environ.copy()
    env["SCRAPY_SETTINGS_MODULE"] = settings_module

    # Create log file
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{spider_name}_{timestamp}.log"

    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour timeout
            )

        if result.returncode == 0:
            return True, f"Completed. Log: {log_file.name}"
        else:
            return False, f"Failed (exit {result.returncode}). Log: {log_file.name}"

    except subprocess.TimeoutExpired:
        return False, f"Timeout after 1 hour. Log: {log_file.name}"
    except Exception as e:
        return False, f"Error: {e}"


@app.command()
def scrape(
    all: bool = typer.Option(False, "--all", help="Run all spiders sequentially"),
    source: str = typer.Option(None, "--source", "-s", help="Specific spider to run"),
    max_pages: int = typer.Option(None, "--max-pages", help="Max pages per spider"),
    max_properties: int = typer.Option(None, "--max-properties", help="Max properties total"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse but don't save to DB"),
):
    """Run property scrapers."""
    if not all and not source:
        console.print("[red]Error:[/red] Must specify --all or --source")
        raise typer.Exit(1)

    # Get spiders to run
    if all:
        spiders_to_run = [s.name for s in get_all_spiders()]
    else:
        if source.lower() not in SPIDERS:
            console.print(f"[red]Error:[/red] Unknown spider: {source}")
            console.print(f"Available: {', '.join(SPIDERS.keys())}")
            raise typer.Exit(1)
        spiders_to_run = [source.lower()]

    console.print(f"\n[bold]Running {len(spiders_to_run)} spider(s)...[/bold]\n")

    results = []
    for spider_name in spiders_to_run:
        config = get_spider(spider_name)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"[cyan]{spider_name}[/cyan]: Running...", total=None)

            success, message = run_spider(
                spider_name,
                max_pages=max_pages or config.default_max_pages,
                max_properties=max_properties,
                dry_run=dry_run,
            )

            progress.remove_task(task)

        status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
        console.print(f"  {status} [bold]{spider_name}[/bold]: {message}")
        results.append((spider_name, success))

    # Summary
    console.print()
    success_count = sum(1 for _, s in results if s)
    fail_count = len(results) - success_count

    if fail_count == 0:
        console.print(f"[bold green]Complete![/bold green] All {success_count} spider(s) succeeded.")
    else:
        console.print(f"[bold yellow]Done.[/bold yellow] {success_count} succeeded, {fail_count} failed.")
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
        cursor.execute("""
            UPDATE listings SET is_active = 0
            WHERE is_active = 1 AND last_seen < ?
        """, (cutoff,))
        conn.commit()
        console.print(f"[green]Marked {count} listings as inactive.[/green]")

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
        table.add_row(
            spider.name,
            spider_type,
            str(spider.default_max_pages),
            spider.description,
        )

    console.print(table)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
