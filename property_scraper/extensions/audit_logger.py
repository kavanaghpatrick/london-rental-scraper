"""
Audit logging extension for Scrapy spiders.

Records comprehensive metrics for each spider run:
- Start/end time, duration
- Items scraped (new, updated, errors)
- HTTP stats (requests, responses, errors)
- Memory usage
- Warnings and errors

Data is stored in SQLite for easy querying and remote monitoring.
"""

import os
import sqlite3
import logging
import time
import psutil
from datetime import datetime
from scrapy import signals
from scrapy.exceptions import NotConfigured

logger = logging.getLogger(__name__)


class AuditLoggerExtension:
    """Scrapy extension that logs comprehensive audit data to SQLite."""

    def __init__(self, db_path, run_id=None):
        self.db_path = db_path
        self.run_id = run_id or datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.conn = None
        self.spider_runs = {}  # spider_name -> run record
        self.process = psutil.Process()

    @classmethod
    def from_crawler(cls, crawler):
        db_path = crawler.settings.get('OUTPUT_DIR', 'output')
        db_path = os.path.join(db_path, 'rentals.db')

        run_id = crawler.settings.get('AUDIT_RUN_ID')

        ext = cls(db_path, run_id)

        # Connect signals
        crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(ext.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(ext.item_scraped, signal=signals.item_scraped)
        crawler.signals.connect(ext.item_dropped, signal=signals.item_dropped)
        crawler.signals.connect(ext.spider_error, signal=signals.spider_error)

        return ext

    def _ensure_tables(self):
        """Create audit tables if they don't exist."""
        cursor = self.conn.cursor()

        # Main scrape runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scrape_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                spider_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                duration_seconds REAL,
                status TEXT DEFAULT 'running',
                items_scraped INTEGER DEFAULT 0,
                items_new INTEGER DEFAULT 0,
                items_updated INTEGER DEFAULT 0,
                items_dropped INTEGER DEFAULT 0,
                items_errors INTEGER DEFAULT 0,
                request_count INTEGER DEFAULT 0,
                response_count INTEGER DEFAULT 0,
                response_bytes INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                memory_start_mb REAL,
                memory_peak_mb REAL,
                memory_end_mb REAL,
                log_file TEXT,
                exit_reason TEXT,
                error_summary TEXT,
                UNIQUE(run_id, spider_name)
            )
        ''')

        # Events table for detailed logging
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scrape_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                spider_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_time TEXT NOT NULL,
                message TEXT,
                details TEXT,
                severity TEXT DEFAULT 'info'
            )
        ''')

        # Indexes for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_runs_run_id ON scrape_runs(run_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_runs_spider ON scrape_runs(spider_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_runs_started ON scrape_runs(started_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_run_id ON scrape_events(run_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON scrape_events(event_type)')

        self.conn.commit()

    def _connect(self):
        """Connect to database and ensure tables exist."""
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path, timeout=60)
            self.conn.execute("PRAGMA journal_mode = WAL")
            self._ensure_tables()

    def _log_event(self, spider_name, event_type, message, details=None, severity='info'):
        """Log an event to the scrape_events table."""
        try:
            self._connect()
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO scrape_events (run_id, spider_name, event_type, event_time, message, details, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (self.run_id, spider_name, event_type, datetime.utcnow().isoformat(),
                  message, details, severity))
            self.conn.commit()
        except Exception as e:
            logger.warning(f"[AUDIT] Failed to log event: {e}")

    def spider_opened(self, spider):
        """Called when spider starts."""
        self._connect()

        now = datetime.utcnow().isoformat()
        memory_mb = self.process.memory_info().rss / 1024 / 1024

        # Get log file path from spider settings
        log_file = getattr(spider, 'log_file', None)

        self.spider_runs[spider.name] = {
            'started_at': now,
            'memory_start_mb': memory_mb,
            'memory_peak_mb': memory_mb,
            'items_scraped': 0,
            'items_dropped': 0,
            'errors': [],
        }

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO scrape_runs (run_id, spider_name, started_at, memory_start_mb, log_file)
            VALUES (?, ?, ?, ?, ?)
        ''', (self.run_id, spider.name, now, memory_mb, log_file))
        self.conn.commit()

        self._log_event(spider.name, 'spider_opened', f"Spider {spider.name} started")
        logger.info(f"[AUDIT] Spider {spider.name} started - run_id: {self.run_id}")

    def spider_closed(self, spider, reason):
        """Called when spider closes."""
        self._connect()

        now = datetime.utcnow().isoformat()
        memory_mb = self.process.memory_info().rss / 1024 / 1024

        run_data = self.spider_runs.get(spider.name, {})
        started_at = run_data.get('started_at')

        # Calculate duration
        duration = None
        if started_at:
            try:
                start_dt = datetime.fromisoformat(started_at)
                end_dt = datetime.fromisoformat(now)
                duration = (end_dt - start_dt).total_seconds()
            except ValueError:
                pass

        # Get stats from spider
        stats = spider.crawler.stats.get_stats()

        items_scraped = stats.get('item_scraped_count', 0)
        items_dropped = stats.get('item_dropped_count', 0)
        request_count = stats.get('downloader/request_count', 0)
        response_count = stats.get('downloader/response_count', 0)
        response_bytes = stats.get('downloader/response_bytes', 0)
        retry_count = stats.get('retry/count', 0)

        # Count errors from stats
        error_count = 0
        for key in stats:
            if key.startswith('spider_exceptions/') or key.startswith('downloader/exception'):
                error_count += stats[key]

        # Determine status
        status = 'completed' if reason == 'finished' else 'failed'

        # Build error summary
        errors = run_data.get('errors', [])
        error_summary = '\n'.join(errors[:10]) if errors else None  # First 10 errors

        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE scrape_runs SET
                finished_at = ?,
                duration_seconds = ?,
                status = ?,
                items_scraped = ?,
                items_dropped = ?,
                request_count = ?,
                response_count = ?,
                response_bytes = ?,
                error_count = ?,
                retry_count = ?,
                memory_peak_mb = ?,
                memory_end_mb = ?,
                exit_reason = ?,
                error_summary = ?
            WHERE run_id = ? AND spider_name = ?
        ''', (now, duration, status, items_scraped, items_dropped,
              request_count, response_count, response_bytes, error_count, retry_count,
              run_data.get('memory_peak_mb', memory_mb), memory_mb,
              reason, error_summary, self.run_id, spider.name))
        self.conn.commit()

        self._log_event(spider.name, 'spider_closed',
                        f"Spider {spider.name} closed: {items_scraped} items, {error_count} errors",
                        details=f"reason={reason}, duration={duration:.1f}s" if duration else f"reason={reason}")

        logger.info(f"[AUDIT] Spider {spider.name} closed - {items_scraped} items, {error_count} errors, reason: {reason}")

    def item_scraped(self, item, spider):
        """Called for each successfully scraped item."""
        if spider.name in self.spider_runs:
            self.spider_runs[spider.name]['items_scraped'] += 1

            # Update peak memory periodically
            if self.spider_runs[spider.name]['items_scraped'] % 100 == 0:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                if memory_mb > self.spider_runs[spider.name].get('memory_peak_mb', 0):
                    self.spider_runs[spider.name]['memory_peak_mb'] = memory_mb

    def item_dropped(self, item, spider, exception):
        """Called when an item is dropped."""
        if spider.name in self.spider_runs:
            self.spider_runs[spider.name]['items_dropped'] += 1

        self._log_event(spider.name, 'item_dropped',
                        f"Item dropped: {str(exception)[:100]}",
                        severity='warning')

    def spider_error(self, failure, response, spider):
        """Called when spider encounters an error."""
        error_msg = str(failure.value)[:200]

        if spider.name in self.spider_runs:
            self.spider_runs[spider.name]['errors'].append(error_msg)

        self._log_event(spider.name, 'spider_error', error_msg,
                        details=f"url={response.url}" if response else None,
                        severity='error')

    def close(self):
        """Clean up database connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
