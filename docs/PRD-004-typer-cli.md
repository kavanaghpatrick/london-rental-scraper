# PRD-004: Typer CLI + Spider Orchestrator

## Overview
Create a unified CLI tool using Typer to orchestrate all spiders and maintenance tasks, replacing the fragmented workflow of running each spider individually.

## Problem Statement
Current workflow requires:
- Running each spider separately: `scrapy crawl savills`, `scrapy crawl rightmove`, etc.
- Manually running post-processing scripts
- No single entry point for "scrape everything"
- Difficult to schedule and automate

## Goals
1. Single CLI entry point: `rental-scraper`
2. Commands: `scrape`, `status`, `mark-inactive`, `dedupe`
3. Spider orchestration with proper Playwright/HTTP mixing
4. Clean console output with progress tracking
5. Cron-friendly for scheduling

## Non-Goals
- Web dashboard (future PRD)
- Distributed scraping (single machine for now)
- Task queuing (simple sequential execution first)

## Technical Design

### CLI Structure

```
rental-scraper scrape [OPTIONS]
    --all               Run all spiders sequentially
    --source <name>     Run specific spider (savills, knightfrank, etc.)
    --max-pages N       Limit pages per spider
    --max-properties N  Limit total properties
    --dry-run           Parse but don't save to DB

rental-scraper status
    Show database stats, recent scrapes, data quality metrics

rental-scraper mark-inactive [OPTIONS]
    --days N            Mark listings not seen in N days as inactive

rental-scraper dedupe [OPTIONS]
    --analyze           Show potential duplicates
    --merge             Merge cross-source duplicates
    --execute           Actually apply changes (default is dry-run)
```

### Spider Registry

```python
# cli/registry.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class SpiderConfig:
    name: str
    spider_class: str
    requires_playwright: bool
    default_max_pages: int
    priority: int  # Lower = run first

SPIDERS = {
    'savills': SpiderConfig(
        name='savills',
        spider_class='property_scraper.spiders.savills_spider.SavillsSpider',
        requires_playwright=True,
        default_max_pages=84,
        priority=1,
    ),
    'knightfrank': SpiderConfig(
        name='knightfrank',
        spider_class='property_scraper.spiders.knightfrank_spider.KnightFrankSpider',
        requires_playwright=True,
        default_max_pages=50,
        priority=2,
    ),
    'chestertons': SpiderConfig(
        name='chestertons',
        spider_class='property_scraper.spiders.chestertons_spider.ChestertonsSpider',
        requires_playwright=True,
        default_max_pages=50,
        priority=3,
    ),
    'foxtons': SpiderConfig(
        name='foxtons',
        spider_class='property_scraper.spiders.foxtons_spider.FoxtonsSpider',
        requires_playwright=False,
        default_max_pages=100,
        priority=4,
    ),
    'rightmove': SpiderConfig(
        name='rightmove',
        spider_class='property_scraper.spiders.rightmove_spider.RightmoveSpider',
        requires_playwright=False,
        default_max_pages=100,
        priority=5,
    ),
}
```

### CLI Implementation

```python
# cli/main.py
import typer
from rich.console import Console
from rich.table import Table
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

app = typer.Typer(help="London Rental Property Scraper CLI")
console = Console()

@app.command()
def scrape(
    all: bool = typer.Option(False, "--all", help="Run all spiders"),
    source: str = typer.Option(None, "--source", "-s", help="Specific spider to run"),
    max_pages: int = typer.Option(None, "--max-pages", help="Max pages per spider"),
    max_properties: int = typer.Option(None, "--max-properties", help="Max properties total"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't save to database"),
):
    """Run property scrapers."""
    if not all and not source:
        console.print("[red]Error:[/red] Must specify --all or --source")
        raise typer.Exit(1)

    spiders_to_run = list(SPIDERS.keys()) if all else [source]

    # Validate spider names
    for spider in spiders_to_run:
        if spider not in SPIDERS:
            console.print(f"[red]Error:[/red] Unknown spider: {spider}")
            raise typer.Exit(1)

    # Sort by priority
    spiders_to_run.sort(key=lambda s: SPIDERS[s].priority)

    console.print(f"[bold]Running {len(spiders_to_run)} spider(s)...[/bold]")

    for spider_name in spiders_to_run:
        config = SPIDERS[spider_name]
        run_spider(config, max_pages, max_properties, dry_run)

    console.print("[bold green]Complete![/bold green]")


def run_spider(config: SpiderConfig, max_pages: int, max_properties: int, dry_run: bool):
    """Run a single spider using CrawlerProcess."""
    console.print(f"\n[cyan]Starting {config.name}...[/cyan]")

    # Choose settings based on spider type
    settings_module = (
        'property_scraper.settings' if config.requires_playwright
        else 'property_scraper.settings_standard'
    )

    settings = get_project_settings()
    settings.setmodule(settings_module)

    # Override settings for dry-run
    if dry_run:
        settings.set('ITEM_PIPELINES', {
            'property_scraper.pipelines.CleanDataPipeline': 100,
            'property_scraper.pipelines.DuplicateFilterPipeline': 200,
        })

    process = CrawlerProcess(settings)

    spider_kwargs = {}
    if max_pages:
        spider_kwargs['max_pages'] = max_pages
    if max_properties:
        spider_kwargs['max_properties'] = max_properties

    process.crawl(config.spider_class, **spider_kwargs)
    process.start()


@app.command()
def status():
    """Show database status and statistics."""
    import sqlite3
    from datetime import datetime, timedelta

    db_path = "output/rentals.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Total listings
    cursor.execute("SELECT COUNT(*) FROM listings")
    total = cursor.fetchone()[0]

    # Active listings
    cursor.execute("SELECT COUNT(*) FROM listings WHERE is_active = 1")
    active = cursor.fetchone()[0]

    # By source
    cursor.execute("""
        SELECT source, COUNT(*),
               SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END),
               MAX(last_seen)
        FROM listings GROUP BY source ORDER BY COUNT(*) DESC
    """)
    by_source = cursor.fetchall()

    # Price history entries
    cursor.execute("SELECT COUNT(*) FROM price_history")
    price_history = cursor.fetchone()[0]

    # Listings with fingerprints
    cursor.execute("SELECT COUNT(*) FROM listings WHERE address_fingerprint IS NOT NULL")
    with_fp = cursor.fetchone()[0]

    # Recent scrapes (last 24h)
    yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
    cursor.execute("SELECT COUNT(*) FROM listings WHERE last_seen > ?", (yesterday,))
    recent = cursor.fetchone()[0]

    conn.close()

    # Display
    console.print("\n[bold]Database Status[/bold]")
    console.print(f"  Total listings: {total:,}")
    console.print(f"  Active listings: {active:,}")
    console.print(f"  With fingerprints: {with_fp:,}")
    console.print(f"  Price history entries: {price_history:,}")
    console.print(f"  Scraped in last 24h: {recent:,}")

    table = Table(title="By Source")
    table.add_column("Source")
    table.add_column("Total", justify="right")
    table.add_column("Active", justify="right")
    table.add_column("Last Seen")

    for source, total, active, last_seen in by_source:
        table.add_row(source, str(total), str(active), last_seen[:10] if last_seen else "N/A")

    console.print(table)


@app.command()
def mark_inactive(
    days: int = typer.Option(7, "--days", "-d", help="Days since last seen"),
    execute: bool = typer.Option(False, "--execute", help="Actually apply changes"),
):
    """Mark listings not seen recently as inactive."""
    import sqlite3
    from datetime import datetime, timedelta

    db_path = "output/rentals.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    # Count affected
    cursor.execute("""
        SELECT COUNT(*) FROM listings
        WHERE is_active = 1 AND last_seen < ?
    """, (cutoff,))
    count = cursor.fetchone()[0]

    console.print(f"Found {count} listings not seen in {days}+ days")

    if not execute:
        console.print("[yellow]Dry run. Use --execute to apply.[/yellow]")
    else:
        cursor.execute("""
            UPDATE listings SET is_active = 0
            WHERE is_active = 1 AND last_seen < ?
        """, (cutoff,))
        conn.commit()
        console.print(f"[green]Marked {count} listings as inactive.[/green]")

    conn.close()


if __name__ == "__main__":
    app()
```

### Package Entry Point

```toml
# pyproject.toml (if using poetry/setuptools)
[project.scripts]
rental-scraper = "cli.main:app"
```

Or directly executable:

```bash
# Install as executable
chmod +x cli/main.py
ln -s $(pwd)/cli/main.py /usr/local/bin/rental-scraper
```

## File Structure

```
scrapy_project/
├── cli/
│   ├── __init__.py
│   ├── main.py          # Typer app entry point
│   ├── registry.py      # Spider registry
│   └── commands/
│       ├── __init__.py
│       ├── scrape.py    # Scrape command
│       ├── status.py    # Status command
│       └── maintenance.py  # mark-inactive, dedupe commands
```

## Dependencies

```
typer[all]>=0.9.0
rich>=13.0.0
```

## Success Criteria

- [ ] `rental-scraper scrape --all` runs all spiders sequentially
- [ ] `rental-scraper scrape --source savills` runs single spider
- [ ] `rental-scraper status` shows database statistics
- [ ] `rental-scraper mark-inactive --days 7` marks stale listings
- [ ] Console output is clean and informative
- [ ] Works from any directory (uses absolute paths)
- [ ] Exit codes: 0 for success, 1 for errors

## Implementation Steps

1. Install typer and rich
2. Create cli/ directory structure
3. Implement spider registry
4. Implement scrape command (basic version)
5. Implement status command
6. Implement mark-inactive command
7. Test all commands
8. Add to pyproject.toml or create wrapper script

## Dependencies
- PRD-001 (Schema must have is_active, first_seen, last_seen)
- PRD-002 (Fingerprint service for status reporting)
- PRD-003 (Pipeline must support new schema)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CrawlerProcess can only run once | Run spiders in separate subprocesses if needed |
| Playwright spiders require different settings | Use settings_module based on spider config |
| Long-running scrapes | Add progress bars, keyboard interrupt handling |
