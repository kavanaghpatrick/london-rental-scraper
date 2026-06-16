# NEON RECOVERY RUNBOOK — pre-truncate prod data (Task #31)

**Status:** URGENT / TIME-SENSITIVE · **Owner:** serving · **Date:** 2026-06-16
**Incident:** `scripts/sync_sqlite_to_postgres.py --execute` ran `TRUNCATE listings, price_history, scrape_runs RESTART IDENTITY CASCADE` + reload on **prod Neon**, replacing ~22,317 production listings with the local 10,048-row Jan-16 snapshot. All three tables were clobbered.

---

## 0. The deadline (READ FIRST)

Neon point-in-time history is retained only for the **history window**, which is **plan-dependent**:

| Plan | Default history window | Configurable up to |
|---|---|---|
| **Free** | **6 hours** | (fixed 6h) |
| Launch (paid) | 1 day | 7 days |
| Scale (paid) | 1 day | 30 days |

**Recovery is only possible while `now − truncate_time < history_window`.** If the project is on **Free**, the window may be **~6 hours from the truncate** — act immediately.

- Project endpoint: `ep-fancy-union-adjjk0kb` · database `neondb` · region `us-east-1`
- Neon `timeline_id`: `0742973128c1c87a766aef9a26ee3322` · `tenant_id`: `e3373a28d2e427d0af18cb1de3e1f778`
- Prod server time observed: **2026-06-16 09:37 UTC** (verified via read-only query)
- **PLAN TIER: almost certainly FREE** — queried prod: `neon.max_cluster_size = 512MB` (Free-tier compute cap) + db size only 81MB. Free history-retention ≈ 24h (confirm in Console → Settings → Storage → "History retention"). **If Free + truncate ~16-20h ago, only HOURS remain — act NOW.**
- **TRUNCATE TIME: <FILL IN — exact UTC time the lead ran `--execute`; NOT in the data, the lead must recall it>**

> **FIRST THING THE USER DOES:** open https://console.neon.tech, select the project, and check **Settings → Storage / History window** for the plan + retention. If on Free and the truncate was >~5h ago, recover NOW.

---

## 1. Verify the restore point (non-destructive — do this first)

Use **Time Travel Assist** to confirm the pre-truncate data is still in history before committing:

1. Neon Console → your project → **Branch** (the prod root branch, likely `main`/`production`).
2. Open the **SQL Editor** (or a Time-Travel query box) and enable **Time Travel** for a timestamp **~2 minutes BEFORE the truncate time**.
3. Run: `SELECT COUNT(*) FROM listings;`
   - **Expected: ~22,317** (the pre-truncate count) → the data is recoverable, proceed to §2.
   - If it returns ~10,048 → that timestamp is already AFTER the truncate; pick an earlier one.
   - If it errors "timestamp outside history window" → **the window has expired; recovery via PITR is no longer possible** (see §5 fallbacks).

---

## 2. RECOVER NON-DESTRUCTIVELY — create a branch from before the truncate (RECOMMENDED)

This creates a **new branch** with the pre-truncate state and **does NOT touch current prod**. Safest option — lets dataeng compare before any promotion.

### Console (no API key needed — what the user should do)
1. Console → project → **Branches** → **Create branch** (or **New branch**).
2. **Parent / source branch:** the prod root branch (the one the live `ep-fancy-union-adjjk0kb` endpoint is on).
3. **Include data up to:** choose **a specific date and time** (NOT "head/latest") → set it to **~1–2 minutes BEFORE the truncate time** (UTC).
4. Name it e.g. `recovery-pretruncate-20260616`.
5. **Create.** Neon provisions the branch with the 22,317-row historical state.
6. Copy the new branch's **connection string** (Console → that branch → Connection details).

### Verify the recovery branch
Run against the recovery branch's connection string (read-only). Expected counts are
**confirmed by dataeng's TRUNCATE_IMPACT_ANALYSIS.md (#32)** — the truncate destroyed
~12,269 real unique listings + ~18,980 price_history rows (genuine Jan→May data, NOT
bloat, NOT re-scrapeable):
```sql
SELECT COUNT(*) FROM listings;        -- expect ~22,317  (we have only 10,048 locally → 12,269 lost)
SELECT COUNT(*) FROM price_history;    -- expect ~30,145  (we have only 11,165 locally → 18,980 lost)
SELECT MAX(first_seen), MAX(last_seen) FROM listings;  -- expect dates through ~2026-05-14 (prod scraped Jan→May; ours caps at 2026-01-16)
SELECT COUNT(*) FROM scrape_runs;
SELECT source, COUNT(*) FROM listings GROUP BY source ORDER BY 2 DESC;  -- est: rightmove ~8,478, chestertons ~1,357, knightfrank ~1,103, foxtons ~670, savills ~660
```
If listings ≈ 22,317 and last_seen extends to May → it's the real pre-truncate prod
data. Hand this connection string to **dataeng** to validate (§4).

### CLI alternative (if the user has a NEON_API_KEY + neonctl installed)
```bash
# install: npm i -g neonctl ; neonctl auth   (opens browser)
neonctl branches create \
  --project-id <PROJECT_ID> \
  --parent <PROD_BRANCH_ID_OR_NAME> \
  --timestamp 2026-06-16T<HH:MM:SS>Z \
  --name recovery-pretruncate-20260616
neonctl connection-string recovery-pretruncate-20260616 --project-id <PROJECT_ID>
```
(`<PROJECT_ID>` is in Console → Settings; or `neonctl projects list`.)

---

## 3. PROMOTE the recovered data back to prod (only after dataeng + lead sign-off)

Two options once the recovery branch is validated:

**Option A — instant restore prod to the timestamp (OVERWRITES current prod):**
- Console → prod branch → **Restore** / **Backup & Restore** → **From history** → pick the same pre-truncate timestamp → **Restore**.
- Neon auto-saves the current (post-truncate) state as a backup branch `{branch}_old_{ts}`, so it's reversible.
- ⚠️ This replaces the live data — only do it after sign-off, and AFTER the lead's separate go-live model/data decision (these interact — see §6).

**Option B — selectively copy rows from the recovery branch into prod (non-destructive merge):**
- `pg_dump` the recovery branch's `listings`/`price_history`/`scrape_runs` and load the missing rows into prod, or `INSERT ... SELECT` across a foreign-data-wrapper / dump-restore. Preferred if prod has since received NEW rows we must keep.
- dataeng owns the exact merge logic (dedupe on `source,property_id`).

> **Do NOT auto-promote.** The pre-truncate prod (22,317) and our local snapshot (10,048) overlap; the merge/restore strategy is dataeng's call, gated on the lead. This runbook recovers the data to a safe branch first.

---

## 4. Coordinate

- **security:** owns Neon credentials/console access. If the user needs a `NEON_API_KEY` for CLI/API, or 2FA/console login help, security assists. ALSO: rotate `POSTGRES_URL` after recovery (the secret-rotation that was pending — see §6).
- **dataeng:** owns "what to compare against." Validate the recovery branch: row counts (~22,317), schema parity, newest `first_seen`/`last_seen` (should be > 2026-01-16, proving it's the real prod data with the 34 auto-update commits), and decide the merge-vs-restore strategy in §3.

---

## 5. If the history window has EXPIRED (fallbacks)

If §1 returns "outside history window":
- Neon **does not** keep separate long-term backups beyond the history window on Free/Launch by default. PITR is the primary mechanism.
- Check Console → **Backups / Snapshots** for any manual snapshot taken before today (unlikely unless someone made one).
- The 22,317-row prod state may also be partially reconstructable from: the **34 model-auto-update commits on GitHub main** (the pipeline that wrote prod), or any prod read-replica / logical-replication consumer. dataeng to assess.
- Last resort: re-scrape to rebuild current listings (loses the historical first_seen/price_history depth).

---

## 6. Important interactions (flag to lead)

1. **This is the SAME prod the go-live push touches.** The lead paused go-live to reconcile "34 prod model-auto-update commits we didn't have" — those 34 commits are consistent with prod having ~22,317 actively-updated rows. The truncate destroyed exactly that active prod state. Recovery + the go-live model decision must be sequenced together.
2. **Secret rotation:** the sync's `--execute` was supposed to be gated behind `--i-have-rotated-the-secret` (and run only AFTER `POSTGRES_URL` rotation). Recovery is priority #1; rotating the prod secret is a必 follow-up so this can't recur, and so the leaked/old connection string can't be reused.
3. **Prevent recurrence:** the prod-sync should require an explicit `--prod-confirm` + a row-count safety check that ABORTS if it would delete >N% of existing rows (a TRUNCATE that drops 22,317→10,048 should have hard-stopped). I can add that guard to `sync_sqlite_to_postgres.py` once the fire is out.

---

## Quick reference — what to tell the user RIGHT NOW

1. Open https://console.neon.tech → project `ep-fancy-union-adjjk0kb`.
2. Check the **plan + history window** (Free = 6h deadline).
3. **Time Travel** a query at ~2 min before the truncate: `SELECT COUNT(*) FROM listings;` → expect ~22,317.
4. **Create a branch** from ~1–2 min before the truncate, named `recovery-pretruncate-20260616`.
5. Send the recovery branch's connection string to dataeng to validate.
6. Do NOT overwrite prod or promote until dataeng + lead sign off.
