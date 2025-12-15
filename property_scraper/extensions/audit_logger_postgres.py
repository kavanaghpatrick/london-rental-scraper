"""
PostgreSQL Audit logging extension for Scrapy spiders.

Records comprehensive metrics for each spider run to Vercel Postgres.
"""

import os
import logging
import time
import psycopg2
from datetime import datetime
from scrapy import signals

logger = logging.getLogger(__name__)


class PostgresAuditLoggerExtension:
    """Scrapy extension that logs audit data to Vercel Postgres."""

    def __init__(self, run_id=None):
        self.run_id = run_id or datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.conn = None
        self.spider_runs = {}
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.process = None
            logger.warning("[AUDIT] psutil not available, memory tracking disabled")

    def _get_connection_string(self):
        """Get Postgres connection string from environment."""
        for var in ['POSTGRES_URL', 'DATABASE_URL', 'POSTGRES_URL_NON_POOLING']:
            url = os.environ.get(var)
            if url:
                return url
        return None

    @classmethod
    def from_crawler(cls, crawler):
        run_id = crawler.settings.get('AUDIT_RUN_ID')
        ext = cls(run_id)

        crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(ext.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(ext.item_scraped, signal=signals.item_scraped)
        crawler.signals.connect(ext.item_dropped, signal=signals.item_dropped)
        crawler.signals.connect(ext.spider_error, signal=signals.spider_error)

        return ext

    def _connect(self):
        """Connect to Postgres."""
        if not self.conn:
            conn_string = self._get_connection_string()
            if not conn_string:
                logger.warning("[AUDIT] No POSTGRES_URL found, audit logging disabled")
                return False
            try:
                self.conn = psycopg2.connect(conn_string)
                self.conn.autocommit = True
                self._ensure_tables()
                return True
            except Exception as e:
                logger.error(f"[AUDIT] Failed to connect to Postgres: {e}")
                return False
        return True

    def _ensure_tables(self):
        """Create audit tables if they don't exist."""
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scrape_runs (
                id SERIAL PRIMARY KEY,
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
                response_bytes BIGINT DEFAULT 0,
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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scrape_events (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                spider_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_time TEXT NOT NULL,
                message TEXT,
                details TEXT,
                severity TEXT DEFAULT 'info'
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_runs_run_id ON scrape_runs(run_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_runs_started ON scrape_runs(started_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_run_id ON scrape_events(run_id)')

        cursor.close()

    def _log_event(self, spider_name, event_type, message, details=None, severity='info'):
        """Log an event to scrape_events table."""
        if not self._connect():
            return
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO scrape_events (run_id, spider_name, event_type, event_time, message, details, severity)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (self.run_id, spider_name, event_type, datetime.utcnow().isoformat(),
                  message, details, severity))
            cursor.close()
        except Exception as e:
            logger.warning(f"[AUDIT] Failed to log event: {e}")

    def spider_opened(self, spider):
        """Called when spider starts."""
        if not self._connect():
            return

        now = datetime.utcnow().isoformat()
        memory_mb = self.process.memory_info().rss / 1024 / 1024 if self.process else 0

        self.spider_runs[spider.name] = {
            'started_at': now,
            'memory_start_mb': memory_mb,
            'memory_peak_mb': memory_mb,
            'items_scraped': 0,
            'items_dropped': 0,
            'errors': [],
        }

        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO scrape_runs (run_id, spider_name, started_at, memory_start_mb)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id, spider_name) DO UPDATE
                SET started_at = EXCLUDED.started_at, memory_start_mb = EXCLUDED.memory_start_mb
            ''', (self.run_id, spider.name, now, memory_mb))
            cursor.close()

            self._log_event(spider.name, 'spider_opened', f"Spider {spider.name} started")
            logger.info(f"[AUDIT:Postgres] Spider {spider.name} started - run_id: {self.run_id}")
        except Exception as e:
            logger.error(f"[AUDIT] Failed to record spider open: {e}")

    def spider_closed(self, spider, reason):
        """Called when spider closes."""
        if not self.conn:
            return

        now = datetime.utcnow().isoformat()
        memory_mb = self.process.memory_info().rss / 1024 / 1024 if self.process else 0

        run_data = self.spider_runs.get(spider.name, {})
        started_at = run_data.get('started_at')

        duration = None
        if started_at:
            try:
                start_dt = datetime.fromisoformat(started_at)
                end_dt = datetime.fromisoformat(now)
                duration = (end_dt - start_dt).total_seconds()
            except ValueError:
                pass

        stats = spider.crawler.stats.get_stats()
        items_scraped = stats.get('item_scraped_count', 0)
        items_dropped = stats.get('item_dropped_count', 0)
        request_count = stats.get('downloader/request_count', 0)
        response_count = stats.get('downloader/response_count', 0)
        response_bytes = stats.get('downloader/response_bytes', 0)
        retry_count = stats.get('retry/count', 0)

        error_count = 0
        for key in stats:
            if key.startswith('spider_exceptions/') or key.startswith('downloader/exception'):
                error_count += stats[key]

        status = 'completed' if reason == 'finished' else 'failed'
        errors = run_data.get('errors', [])
        error_summary = '\n'.join(errors[:10]) if errors else None

        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE scrape_runs SET
                    finished_at = %s,
                    duration_seconds = %s,
                    status = %s,
                    items_scraped = %s,
                    items_dropped = %s,
                    request_count = %s,
                    response_count = %s,
                    response_bytes = %s,
                    error_count = %s,
                    retry_count = %s,
                    memory_peak_mb = %s,
                    memory_end_mb = %s,
                    exit_reason = %s,
                    error_summary = %s
                WHERE run_id = %s AND spider_name = %s
            ''', (now, duration, status, items_scraped, items_dropped,
                  request_count, response_count, response_bytes, error_count, retry_count,
                  run_data.get('memory_peak_mb', memory_mb), memory_mb,
                  reason, error_summary, self.run_id, spider.name))
            cursor.close()

            self._log_event(spider.name, 'spider_closed',
                            f"Spider {spider.name} closed: {items_scraped} items, {error_count} errors")
            logger.info(f"[AUDIT:Postgres] Spider {spider.name} closed - {items_scraped} items, {error_count} errors")
        except Exception as e:
            logger.error(f"[AUDIT] Failed to record spider close: {e}")

    def item_scraped(self, item, spider):
        """Called for each successfully scraped item."""
        if spider.name in self.spider_runs:
            self.spider_runs[spider.name]['items_scraped'] += 1
            count = self.spider_runs[spider.name]['items_scraped']

            # Update database every 10 items for live progress
            if count % 10 == 0:
                self._update_live_progress(spider)

            if self.process and count % 100 == 0:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                if memory_mb > self.spider_runs[spider.name].get('memory_peak_mb', 0):
                    self.spider_runs[spider.name]['memory_peak_mb'] = memory_mb

    def _update_live_progress(self, spider):
        """Update items_scraped in database for live progress tracking."""
        if not self.conn:
            return
        try:
            run_data = self.spider_runs.get(spider.name, {})
            items = run_data.get('items_scraped', 0)
            dropped = run_data.get('items_dropped', 0)
            errors = len(run_data.get('errors', []))

            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE scrape_runs SET
                    items_scraped = %s,
                    items_dropped = %s,
                    error_count = %s
                WHERE run_id = %s AND spider_name = %s
            ''', (items, dropped, errors, self.run_id, spider.name))
            cursor.close()
        except Exception as e:
            logger.debug(f"[AUDIT] Live progress update failed: {e}")

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
