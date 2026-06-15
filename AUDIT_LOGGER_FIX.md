# AUDIT LOGGER FIX — scrape-run finalization hardening

**Task:** #20 — Harden scrape-run finalization (zombie `status='running'` root cause)
**Owner:** `dataeng` (audit logger extensions are the run-lifecycle writers; isolated changes only)
**Root cause diagnosis:** `scraper-kf-ch`
**Date:** 2026-06-15

---

## Symptom

`scrape_runs.id=4` (knightfrank, run `20260114_170343`) sat at `status='running'` from Jan 14 onward. The dashboard `/api/running` reports any `running` row as a live spider, so this dead run showed as a forever-running spider. I hand-cleaned that specific row earlier (task #2); this task makes the class of bug **self-healing**.

## Root cause (NOT a spider bug)

The run lifecycle is owned by the audit logger **extensions**, not the spiders:

- `property_scraper/extensions/audit_logger.py` — **SQLite** (used by `settings.py`, `settings_standard.py`)
- `property_scraper/extensions/audit_logger_postgres.py` — **Postgres** (used by `settings_postgres.py`, `settings_postgres_standard.py`)

Both `INSERT` a row with `status='running'` on `spider_opened`, and only `UPDATE` it to `completed`/`failed` on `spider_closed`.

Scrapy's `spider_closed` fires on **normal finish, raised exceptions, AND `CloseSpider`/timeout** (`closespider_timeout`, `closespider_errorcount`, etc.). The close handler already sets `status = 'completed' if reason == 'finished' else 'failed'` and writes `finished_at` + `exit_reason`. **So crashes, errors, and timeouts were already finalized correctly** — verified empirically by scraper-kf-ch (clean close) and by the test below (timeout close).

The ONLY way a row stays `running` is **hard external process death** before `spider_closed` can run:
`SIGKILL` / `kill -9` / OOM-killer / power loss / container eviction / reboot. No in-process signal handler can finalize a row when the process is hard-killed — by definition no Python code runs at that point.

## Fix — startup reconciliation (the reaper)

Added `_reap_stale_runs()` to **both** loggers, called on `spider_opened` **before** the new row is inserted:

```sql
UPDATE scrape_runs
SET status='failed', finished_at=<now>, exit_reason='stale_orphan',
    error_summary='Auto-reaped ... >Nh (process killed before spider_closed) ...'
WHERE status='running' AND run_id != <current_run_id> AND started_at < <now - Nh>
```

Properties:
- **Threshold `N` = `AUDIT_STALE_RUN_HOURS` (default 4h)** — deliberately above `CLOSESPIDER_TIMEOUT` (3600s = 1h) so a legitimately long-running spider is never falsely reaped. Set `<=0` to disable.
- **Excludes the current `run_id`** so concurrent spiders sharing one process/run are never touched.
- **Non-fatal**: any error in the reap is caught and logged as a warning; it never blocks a scrape.
- **Self-healing**: the next scrape after any hard-kill automatically finalizes the orphan, so `/api/running` and `active%` stop lying without manual `sqlite3`/`psql` surgery.

This is the correct and complete remedy. An in-process `SIGTERM`/atexit handler was considered but rejected as insufficient: it cannot cover `SIGKILL`/OOM/reboot (the actual cause), and Scrapy already converts a graceful `SIGTERM` into a normal shutdown that fires `spider_closed`. The reaper covers every abnormal-death path uniformly.

## Exit-path coverage matrix

| Exit path | `spider_closed` fires? | Finalized as | By |
|---|---|---|---|
| Normal finish | yes (`reason='finished'`) | `completed` | `spider_closed` |
| Raised exception in spider | yes | `failed` | `spider_closed` |
| `closespider_timeout` (>1h) | yes (`reason='closespider_timeout'`) | `failed` | `spider_closed` |
| `closespider_errorcount` | yes | `failed` | `spider_closed` |
| Graceful `SIGTERM` (single `kill`) | yes (Scrapy graceful shutdown) | `failed` | `spider_closed` |
| **`SIGKILL` / `kill -9` / OOM / reboot** | **NO** | `failed` / `stale_orphan` | **reaper on next `spider_opened`** |

## Verification (real extension, not mocks of the SUT)

Ran the actual `AuditLoggerExtension` (SQLite) against a temp DB — canonical untouched:

- **PASS A** — Simulated hard-kill: `spider_opened` inserts `running`, then NO `spider_closed` (process death), started_at backdated 10h. A subsequent `spider_opened` on a new run reaped it → `status='failed'`, `exit_reason='stale_orphan'`, `finished_at` set. The concurrent/current run stayed `running` (not falsely reaped).
- **PASS B** — `spider_closed(reason='closespider_timeout')` finalized the row as `failed` with `exit_reason='closespider_timeout'` and `finished_at` set, confirming the timeout/abnormal-close path was already correct.
- Final state: **0 stale `running` rows**.

Postgres logger mirrors the SQLite logic exactly (`%s` params, `psycopg2`), so the verified behavior carries over; it requires a live `POSTGRES_URL` to exercise end-to-end (left for serving #8 / automation #9 in their env).

## Files changed (isolated to the audit logger extensions)

- `property_scraper/extensions/audit_logger.py` — added `_reap_stale_runs()` + call in `spider_opened`; `STALE_RUN_HOURS`/`AUDIT_STALE_RUN_HOURS` config; `timedelta` import.
- `property_scraper/extensions/audit_logger_postgres.py` — same.

No spider, pipeline, or schema changes. No change to the canonical `output/rentals.db` data.

## Coordination notes

- **automation (#9):** the reaper runs automatically at the start of every scrape via `spider_opened`. If you want a *standalone* sweep (e.g. a cron/CI step or a `cli.main` subcommand independent of a scrape), ask — it's a 6-line wrapper around the same UPDATE. Not added now to keep the change isolated per the task.
- **serving (#8):** `/api/running` will now self-correct after the first post-deploy scrape; no dashboard change needed. If you want immediate correction without waiting for a scrape, the standalone sweep above is the hook.
- Tunable: set `AUDIT_STALE_RUN_HOURS` in settings if max expected runtime ever exceeds 4h.
