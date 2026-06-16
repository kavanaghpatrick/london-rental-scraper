# BACKUP & DATA-SAFETY STRATEGY

**Owners:** `security` (cloud / Neon) + `dataeng` (local SQLite). Task #38.
**Status:** Cloud side implemented + statically verified. Local side: see §5/§6 (dataeng).
**Trigger:** On 2026-06-16 a prod `TRUNCATE` (old sync) wiped Neon from ~22,317 → 10,048
rows. **Recovery FAILED** — the project is Neon **Free tier (6-hour PITR window)** and
the truncate was ~18h before recovery was attempted, with no snapshots. **~12,269
listings + ~18,980 price-history rows are permanently lost.** This regime is now the
SOLE mechanism ensuring it can never recur.

> ## ⭐ TOP RECOMMENDATION TO THE USER — upgrade Neon to **Launch** (cheap insurance)
> The data loss happened **because** Free tier gives only a **6-hour** PITR window.
> **Neon Launch raises the restorable history window to 7 days** (Scale: 30 days) for
> a low monthly cost. With a 7-day window, an accidental write like this one is
> recoverable for a week instead of 6 hours. **This single change would have fully
> prevented the loss.** Strongly recommend the user upgrade to at least Launch and set
> the history window to its max (7 days). The dump-based layers below are the backstop;
> a real PITR window is the front-line defense. *(Lead is surfacing this to the user.)*

> Standing constraint (lead's order): NO live-Neon operations until the go-live sync.
> The first real prod dump runs as part of **re-populating prod post-rebuild** (after
> merge+retrain, via serving's now-safe UPSERT sync — which already dumps first). Rotate
> `POSTGRES_URL` per `dashboard/ROTATION_RUNBOOK.md` before that write.

---

## 0. Defense in depth — the layers

| Layer | Protects against | Where | Owner |
|---|---|---|---|
| **L1. Neon PITR** (instant restore) | accidental prod writes within the history window | Neon console | security |
| **L2. Scheduled prod dump** | total prod loss, drift, point-in-time beyond PITR window | `.github/workflows/neon-backup.yml` → CI artifact | security |
| **L3. Pre-destructive-op prod dump** | a bad sync nuking prod | `scripts/backup_neon.sh` — **wired into the UPSERT sync, runs before any write** | security |
| **L4. Per-op row backup + abort guard** | dedupe clustering bug over-deleting | `daily-scrape.yml` dedupe step (CSV of doomed rows + 10% abort) | automation/#37 |
| **L5. Local pre-op DB snapshot** | destructive local pipeline stages | `automation/daily_pipeline.py` `_backup_database()` (verified; 2 gaps in §5) | dataeng |
| **L6. Golden master** | "lost the merged complete dataset" | `scripts/archive_golden_master.py` → `golden_master/` (read-only sibling of `output/`) | dataeng + security |

No single layer is sufficient; the loss happened because L2/L3 did not exist.

---

## 1. Neon PITR (L1) — confirm/maximize retention  *(USER action)*

Neon "instant restore" rolls a branch back to any point inside the project's
**history window**. Retention by plan:

| Plan | Default window | **Max window** |
|---|---|---|
| Free (Hobby) | 6 hours | **6 hours** |
| Launch | 1 day | **7 days** |
| Scale | 1 day | **30 days** |

**This project is on the Free/Hobby tier** (confirmed: Vercel OIDC `"plan":"hobby"`,
and the user verified the 6h window in the console). That 6-hour window is **exactly
why the truncate was unrecoverable** — see the top recommendation: **upgrading to
Launch (7-day window) is the single highest-value fix.** Until/unless upgraded, the
L2 scheduled dump is the primary recovery mechanism (it has no time limit).

**USER steps:**
1. Neon console → project → **Settings → Storage / History retention**.
2. Set the history window to the **max your plan allows** (Free: already 6h, no
   knob; Launch: 7d; Scale: 30d).
3. **If you want >6h PITR, a plan upgrade is required** (Launch→7d or Scale→30d).
   Flag: decide whether the data's value justifies the upgrade. Until then, the
   scheduled dump (L2, below) is the primary recovery mechanism — it has no such
   limit.
4. To restore: create a **restore branch** at the target timestamp for analysis,
   or roll the branch back. Never overwrite prod in place without a fresh dump first.

---

## 2. Scheduled prod dump (L2) — `.github/workflows/neon-backup.yml`

Daily read-only `pg_dump` of prod Neon, uploaded as a CI artifact.

- **Schedule:** `17 5 * * *` UTC — 43 min before the 06:00 daily scrape, so a
  clean pre-scrape snapshot always exists.
- **Read-only:** `pg_dump` never writes to the DB.
- **Gated:** the job fails fast if the `POSTGRES_URL` repo secret is unset.
- **Retention:** CI artifact kept **90 days** (independent of the script's local prune).
- **Manual run:** Actions → "Neon Prod Backup" → *Run workflow*.

**Restore from an L2 artifact:**
```bash
# download the artifact, then:
gunzip -c neon_YYYYMMDD_HHMMSSZ.sql.gz | psql "$POSTGRES_URL_OF_A_RESTORE_BRANCH"
# Restore into a NEW Neon branch first, verify counts, then promote. Never pipe
# straight into prod.
```

---

## 3. Pre-destructive-op prod dump (L3) — `scripts/backup_neon.sh`

A versioned, compressed, **fail-closed** prod dump. Take one immediately before
ANY destructive prod op.

- Output: `output/backups/neon/neon_<UTC-timestamp>.sql.gz`.
- **Fail-closed:** exits **non-zero** if `pg_dump` is missing, the URL is unset, or
  the dump is empty/invalid — so a caller can abort the destructive op.
- Redacts credentials in all logs. Handles the Vercel trailing-`\n` URL artifact.
- **Auto-rewrites the Neon `-pooler` endpoint → direct** (idempotent). `pg_dump`
  cannot use the pgbouncer pooler (transaction pooling drops the session state it
  needs). The sync already rewrites before calling the script; the script also does
  it so **standalone/scheduled runs** (`neon-backup.yml`, manual) work too.
- Prunes dumps older than `RETENTION_DAYS` (default 30) but **never the newest**.
- Requires the PostgreSQL 16 client:
  - macOS: `brew install libpq && brew link --force libpq`
  - Debian/CI: `sudo apt-get install -y postgresql-client`

**WIRED (2026-06-16):** the prod sync (`scripts/sync_sqlite_to_postgres.py`, now
**non-destructive UPSERT** since #37 — no more TRUNCATE) calls `backup_neon.sh`
**in-process before any write** via `subprocess.run([..., "backup_neon.sh"], check=True)`.
Because the script is fail-closed, `check=True` raises and **the load never runs
without a verified dump on disk.** The sync even REFUSES to write if `backup_neon.sh`
is missing (unless `--skip-pg-dump` is explicitly passed). So the L3 guard is
automatic — no separate caller step needed:
```bash
# 1. (USER) rotate POSTGRES_URL first — ROTATION_RUNBOOK.md
# 2. run the gated sync; it dumps prod (backup_neon.sh) BEFORE writing, aborts if the dump fails
python3 scripts/sync_sqlite_to_postgres.py --execute --i-have-rotated-the-secret
```
The sync also adds an in-transaction JSON snapshot + a row-count safety guard
(an UPSERT must never reduce a table's row count) as a second net. `backup_neon.sh`
remains the authoritative, restorable backup.

**Preflight without dumping:** `scripts/backup_neon.sh --check`.

---

## 4. Per-op row backup + abort guard (L4) — already in `daily-scrape.yml`

The daily-scrape dedupe step (added under #37) already:
- **Aborts** if the DELETE would remove >10% of active listings (`MAX_DELETE_FRACTION`).
- **Backs up** every to-be-deleted row to `dedupe_deleted_backup_<ts>.csv`
  (uploaded as an artifact) before deleting.

This is row-level and complements — does not replace — the full-DB L2/L3 dumps.
Recommended hardening (for #40/automation owner): also run `scripts/backup_neon.sh`
once at the **start** of the daily-scrape job so a full snapshot precedes *all*
that run's prod mutations (mark-inactive UPDATE + dedupe DELETE), not just the
dedupe DELETE's own rows.

---

## 5. Local SQLite versioning (L5)  — verified by dataeng (2026-06-16)

The canonical store is `scrapy_project/output/rentals.db` (sole writer: `dataeng`,
per `DATA_LAYER_CONTRACT.md`). `automation/daily_pipeline.py` → `_backup_database()`
gzips the DB to `output/backups/rentals_<ts>.db.gz`.

**VERIFIED PRESENT (good):** the backup runs in preflight (step 4) **before**
mark-inactive (step 5) and the dedupe stage, after an integrity check. So
backup-before-destructive exists.

**3 GAPS found by dataeng (fixes touch `automation/daily_pipeline.py`, which is
AUTOMATION's file — dataeng may verify freely but hardening edits need automation's
ack; being routed to them):**

- **GAP A — backup not integrity-verified after gzip.** A readable-but-corrupt
  `.db.gz` would pass silently. Fix: `gzip -t` + a `PRAGMA integrity_check` on the
  compressed copy (same validation `backup_neon.sh` does on its dumps).
- **GAP B — retention has NO count floor (REAL RISK).** `_cleanup_old_backups()`
  deletes ALL `*.db.gz` older than `keep_backups_days` (default 7) with no minimum.
  In the **frozen-snapshot situation** (pipeline idle 7+ days — exactly this
  project's state), the next run purges **every** backup → **zero copies**. Fix:
  "always keep the newest N regardless of age" floor — the same rule already in
  `backup_neon.sh` §3.
- **GAP C — loose `.bak` files in the destructive path.** Pre-existing one-off
  backups (`rentals.db.pre_dataeng_*.bak`, `rentals.db.damaged_allinactive_*.bak`,
  `rentals_merged.db`) sit unmanaged inside `output/`. The golden master (§6)
  covers the keep-safe need; cleaning these loose ones is janitor/#33/#1 territory.

**Local restore:** `gunzip -c output/backups/rentals_<ts>.db.gz > output/rentals.db`
(stop the pipeline first; verify with `sqlite3 output/rentals.db 'PRAGMA integrity_check;'`).

---

## 6. Golden master (L6) — immutable recovered dataset

**Local (dataeng) — DONE:** `scripts/archive_golden_master.py` writes a
timestamped, integrity-verified (source AND copy, with sha256 match), **read-only
(0444)** copy into **`golden_master/`** — a **SIBLING of `output/`**, so the
pipeline's age-based backup cleanup (§5 GAP B) can never reach it. Each archive has
a sidecar manifest (row counts + sha256 + provenance). Overwrite of an existing
golden file is blocked by design. First archive of the canonical DB:
`golden_master/rentals_golden_canonical_20260616_114253.db` (10,048 / 7,763 active /
11,165 price_history). dataeng will **re-run it on the MERGED dataset** once PITR
recovery + the union (#36) land — that becomes the true "everything, once, immutable" anchor.

**Cloud (security):** mirror the same merged golden master as a dedicated, retained
artifact / a one-off `pg_dump` tagged `golden`, kept **separate** from the rolling
L2 dumps so retention pruning can never reap it. Gated on the merged dataset existing
+ secret rotation.

---

## 7. What's done vs gated

| Item | State |
|---|---|
| `scripts/backup_neon.sh` (L3) | **Done.** Syntax + shellcheck clean; full dry-run tested with a fake `pg_dump` (success, failure, empty-dump, no-URL paths all correct); no live Neon contact. |
| `.github/workflows/neon-backup.yml` (L2) | **Done.** Valid YAML. First real run gated on the `POSTGRES_URL` secret existing (post-rotation). |
| PITR guidance (L1) | **Documented.** ⭐ Recommend Neon **Launch upgrade (7-day window)** — lead surfacing to user. Retention change/upgrade is the USER's. |
| Serving-sync pre-op guard wiring (L3↔sync) | **DONE / WIRED.** `sync_sqlite_to_postgres.py` (#37, non-destructive UPSERT) calls `backup_neon.sh` in-process before any write (`check=True`, refuses to write if missing). |
| Local backup-before-destructive (L5) | **Verified present** by dataeng. |
| Local retention floor (L5 GAP B) | **Known issue, fix routed to automation** — frozen-snapshot purge risk. |
| Local backup integrity check (L5 GAP A) | **Known issue, fix routed to automation.** |
| Golden master, local (L6) | **DONE** — `scripts/archive_golden_master.py` → `golden_master/` (read-only, manifest). Re-run on merged data post-recovery. |
| Golden master, cloud (L6) | **Gated** on merged dataset + rotation. |
| First real prod dump | **Runs automatically at go-live** — the UPSERT sync dumps prod before re-populating it post-rebuild. |

## 8. Go-live checklist (when re-populating prod after merge + retrain)
1. ⭐ **Recommend upgrading Neon to Launch (7-day PITR)** before go-live — cheap insurance, would have prevented the loss.
2. (USER) rotate `POSTGRES_URL` — `dashboard/ROTATION_RUNBOOK.md`.
3. Ensure `pg_dump` is available (CI installs it; locally `brew install libpq`).
4. Run the gated UPSERT sync: `python3 scripts/sync_sqlite_to_postgres.py --execute --i-have-rotated-the-secret`.
   → it runs `backup_neon.sh` (first real prod dump) BEFORE writing, and aborts if that dump fails.
5. Set the GitHub secret `POSTGRES_URL` (rotated) so `neon-backup.yml` runs nightly thereafter.
6. Re-run `scripts/archive_golden_master.py` on the merged dataset, and mirror it to a cloud `golden` dump.

> Backup scripts are exercised in dataeng's test suite (#46, non-destructive sync) and
> wired into CI by automation (#49).
