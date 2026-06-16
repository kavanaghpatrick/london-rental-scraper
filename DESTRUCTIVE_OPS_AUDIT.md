# DESTRUCTIVE OPS AUDIT (#37 / #40)

**Author:** `automation`
**Date:** 2026-06-16
**Scope:** automation's half of #37 — audit every data-writing path for destructive
operations (TRUNCATE / DELETE / destructive UPDATE) and gate them behind a backup +
row-count-delta safety check. (The prod-sync UPSERT + pg_dump is serving's half.)

---

## Root cause

`scripts/sync_sqlite_to_postgres.py` used `TRUNCATE ... RESTART IDENTITY CASCADE`
then bulk-reloaded from local SQLite. When local had fewer rows than prod (10,048 vs
22,317), the TRUNCATE destroyed the prod-only rows. **Lesson:** no data-writing path
should be able to delete a large slice of data without (a) a delta guard and (b) a
recoverable backup taken first.

---

## Full inventory of destructive ops (core data = `listings` / `price_history`)

| # | Location | Op | Auto-runs? | Owner | Status |
|---|----------|----|-----------|-------|--------|
| 1 | `scripts/sync_sqlite_to_postgres.py` | was `TRUNCATE` (prod) | gated (--execute) | serving | **FIXED (serving)** — TRUNCATE→UPSERT (ON CONFLICT DO UPDATE/NOTHING), no setval, prod-only rows PRESERVED, fail-closed pg_dump + in-txn JSON snapshot before write, >5% row-drop abort. Code-reviewed (5 bugs fixed, see below). |
| 2 | `.github/workflows/daily-scrape.yml` "Clean duplicate listings" | `DELETE FROM listings/price_history` (prod) | **YES, every scrape, was ungated** | automation | **FIXED** — delta-guard + backup |
| 3 | `dedupe_cross_source.py:remove_duplicates` | `DELETE FROM listings` (local) | no (manual CLI) | automation | **FIXED** — guarded_delete |
| 4 | `scripts/dedupe_same_source.py:execute_deduplication` | `DELETE FROM listings/price_history` (local) | no (manual CLI) | automation | **FIXED** — guarded_delete |
| 5 | `automation/daily_pipeline.py` mark-inactive | `UPDATE is_active=0` | yes (preflight) | automation | already triple-guarded (#25): frozen-snapshot + cycle-relative + >50% abort |
| 6 | `cli/main.py` mark-inactive | `UPDATE is_active=0` | no (manual CLI) | automation | already triple-guarded (#25) + `--force` |

### Archived / dead (no longer a live path)
- `merge_and_delete_duplicates.py` (named in the task) was archived by janitor to
  `archive/_cleanup_2026-06-15/dead_scripts/`. It is NOT auto-run by any pipeline/CI
  and is dead code — no guard added (would be archiving a guard onto a dead script).

### Non-destructive (audited, no guard needed)
- `automation/daily_pipeline.py::_run_dedupe` and `cli/main.py::dedupe` — only
  `UPDATE listings SET size_sqft = ...` (ADDITIVE: fills missing sqft from agent
  sources, never deletes). Safe.
- `dedupe.py:366,414` — `DROP TABLE IF EXISTS duplicate_groups / dedupe_stats`
  immediately followed by `CREATE TABLE` of the same scratch/analysis tables. These
  are NOT core data (`listings`/`price_history`); they're recomputed each run. Not a
  data-loss vector — noted, no guard added.

---

## The shared guard: `scripts/_safe_delete.py`

One reusable `guarded_delete()` so every DELETE path enforces the same invariant:

1. **DELTA GUARD** — abort (`SafeDeleteAborted`) if `len(ids) / total_rows` exceeds
   `max_fraction` (default **10%**). A correct dedupe retires a small tail; a >10%
   delete means a bug. Nothing is deleted.
2. **BACKUP** — dump the to-be-deleted rows (full columns) to a timestamped CSV under
   `output/deleted_backups/` BEFORE deleting. Over-deletion is recoverable.
3. **DELETE** — only then run the caller's delete callback (caller owns txn/paramstyle).

Works for both sqlite3 and psycopg2 cursors. Verified by unit test: a 5% delete
proceeds + backs up; a 52% delete ABORTS with nothing deleted.

### CI dedupe (#2) — special-cased inline
The daily-scrape "Clean duplicate listings" step runs against PROD inside an inline
`python3 <<EOF` heredoc, so it can't `import` the helper. It gets the equivalent
inline guard: abort if `>10%` of active listings would be deleted (`::error::` +
non-zero exit), and a CSV backup of the doomed rows written + uploaded as the
`scrape-logs` artifact (added `dedupe_deleted_backup_*.csv` to the upload paths).

---

## Files changed (automation, #37)

| File | Change |
|------|--------|
| `scripts/_safe_delete.py` | NEW — `guarded_delete()` + `SafeDeleteAborted` |
| `.github/workflows/daily-scrape.yml` | dedupe DELETE: inline >10% delta-abort + CSV backup; backup added to artifact upload |
| `dedupe_cross_source.py` | `--remove` DELETE routed through `guarded_delete()` |
| `scripts/dedupe_same_source.py` | `execute_deduplication` batches all deletes through `guarded_delete()` |
| `.gitignore` | ignore `output/deleted_backups/` (recovery artifacts, never commit) |

All uncommitted, staged for the lead's batch.

---

## Sync (serving's half) — FIXED + code-reviewed

`scripts/sync_sqlite_to_postgres.py` is now non-destructive (serving, owner). TRUNCATE
gone → `INSERT ... ON CONFLICT (business key) DO UPDATE` (listings/scrape_runs) / `DO
NOTHING` (price_history, append-only); no setval; prod-only rows preserved;
`price_history.listing_id` remapped (SQLite id → (source,property_id) → prod id);
`ensure_pg_parity` adds the `UNIQUE(listing_id, recorded_at)` constraint the ON CONFLICT
target needs; fail-closed pg_dump + in-txn JSON snapshot before write; >5% row-drop abort.

**5 bugs serving's review caught + fixed:** (1) pg_dump given the `-pooler` URL → fails on
pgbouncer → would abort every sync; now strips `-pooler` + 600s timeout. (2) `canonical_id`
copied verbatim (self-ref to `listings.id`) → corrupt in prod; now NULLed. (3) `first_seen`
clobbered on DO UPDATE → now excluded (set-once contract). (4) partial conflict key bypassed
the refuse-guard → now refuses on partial too. (5) [serving's 5th].

**2 residual items serving handed to this audit (latent, lower severity):**
- The >5% row-count guard is ADDITIVE-ONLY (UPSERT can't shrink), so it can't catch
  loss-masked-by-inserts or remap-skips. Better: assert `price_history.processed > 0` and
  skip-count ≈ 0, rather than relying on a net row-count delta. (serving to decide.)
- `NULL recorded_at` defeats `ON CONFLICT` (Postgres treats NULLs as distinct) → unbounded
  re-insert of those history rows. 0 NULL rows currently, but add a guard if prod ever has
  them. (serving to decide.)

These two are in serving's file; flagged here for completeness, not fixed by automation.

## Handoffs / still open

- **serving** owns the prod-sync fix (above) — DONE + reviewed. Resolved.
- Semantics note: UPSERT means prod-only rows now PERSIST (the whole point — never lose
  data). If the dashboard needs stale rows hidden, that's a separate `is_active`/staleness
  pass, NOT deletion.
- The `max_fraction=10%` threshold is conservative; tune per-op if a legitimate large
  dedupe is ever expected (none currently is).
