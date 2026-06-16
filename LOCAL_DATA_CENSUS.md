# LOCAL DATA CENSUS — 2026-06-16 (task #33)

Read-only inventory of EVERY local data store. **No DB was modified** (all reads via
`sqlite3 ... mode=ro`; `.gz` backups gunzipped to temp files, then deleted).

> **⚠️ STALE — numbers below are a snapshot as of 2026-06-16 ~10:25–11:02 and are NOW SUPERSEDED.**
> Events since: the robust re-scrape completed, `merge_datasets.py` RAN (#42 done), and
> **canonical `output/rentals.db` is now ~14,836 listings** (not 10,048). dataeng also fixed two
> post-merge defects (restored 9 dropped audit/dedupe tables from golden master; deleted a
> bad-price £0 row). A scrape is **still running** (active WAL; counts moving 14,453→14,836), so
> canonical remains a moving target. **A full census refresh is PENDING** — on hold per dataeng
> until the scrape quiesces and the final cycle-relative `is_active` recompute is done; dataeng
> will ping with the frozen baseline, then this file gets re-run. The *structural* conclusions
> (no local data beyond what's now merged into canonical; the prod-only gap is PITR/git/artifact
> territory; redundant `.gz`; decoy/empty DBs) hold regardless of the row-count refresh.

## TL;DR for the recovery effort (#31/#32/#36/#39)

- **Local universe = exactly 10,048 distinct `(source, property_id)` listings, all dated ≤ 2026-01-16 10:03.** Nothing local extends past Jan-16. So prod's 22,317 implies **~12,269 listings that exist ONLY in prod** — NOT recoverable from any local DB/backup/jsonl here.
- **MOST COMPLETE local store (by ROW/HISTORY) = canonical `output/rentals.db` (= all 8 `.gz` + `pre_dataeng.bak`):** 10,048 rows + 11,165 price_history. The `pre_dataeng.bak` has 1,122 MORE *active* rows (8,885 vs 7,763), but — **CORRECTION (per dataeng, verified)** — those 1,122 are NOT recoverable signal: they are the exact rows dataeng's `mark-inactive` (task #2) correctly deactivated. I independently confirmed **all 1,122 have `last_seen` > 7 days before max** (1122/1122 stale), i.e. genuinely not-seen-in-final-cycle. Canonical's `is_active=7763` is the CORRECT cleaned state; the .bak is the pre-clean state. **Do NOT source `is_active` from the .bak** — that would re-activate stale listings and break serving's "what's live" view. The union (#36) recomputes `is_active` cycle-relative post-merge (source-independent → correct 7,763). So there is effectively **no local "data we think is gone"** beyond what's already in canonical.
- **No local backup predates the dataset** in a useful way: the oldest full snapshots (Dec 2025) have FEWER rows (5,030 / 2,413 / etc.) — they're earlier in the scrape history, not a richer pre-loss copy. The Jan-16 ceiling is the same everywhere.
- All 8 `output/backups/*.db.gz` are **byte-identical** to canonical after gunzip (verified sha256) — redundant Jun-15 snapshots of the same 10,048/active=7763 state.

## Full table: file → rows → date range → per-source → notes

| File | listings | price_history | last_seen range | per-source (listings) | notes |
|------|---------:|--------------:|-----------------|------------------------|-------|
| **output/rentals.db** (CANONICAL) | 10,048 | 11,165 | 2025-12-06 → **2026-01-16** | rm 6933, ch 1110, kf 902, fx 548, sv 540, jdw 15 | **active=7763**. The live canonical. |
| **output/rentals.db.pre_dataeng_20260615_150228.bak** | 10,048 | 11,165 | 2025-12-06 → 2026-01-16 | identical sources | **active=8885 (+1,122 vs canonical)** — MOST COMPLETE by active count. Pre-dataeng-inactivation snapshot. |
| output/rentals.db.damaged_allinactive_20260615_152403.bak | 10,048 | 11,165 | 2025-12-06 → 2026-01-16 | identical sources | **active=0** — DAMAGED (wall-clock-wipe, task #25). Same rows, all inactivated. Do NOT use. |
| output/backups/rentals_20251209.db | 5,030 | 7,678 | 2025-12-06 → **2025-12-09** | rm 3041, kf 710, ch 655, sv 397, fx 212, jdw 15 | active=5030. Earlier scrape state (Dec-9); fewer rows. |
| output/backups/rentals_20260615_151617.db.gz | 10,048 | 11,165 | → 2026-01-16 | identical | active=7763. = canonical (redundant). |
| output/backups/rentals_20260615_152558.db.gz | 10,048 | 11,165 | → 2026-01-16 | identical | active=7763. byte-identical to canonical. |
| output/backups/rentals_20260615_152645.db.gz | 10,048 | 11,165 | → 2026-01-16 | identical | active=7763. byte-identical. |
| output/backups/rentals_20260615_153612.db.gz | 10,048 | 11,165 | → 2026-01-16 | identical | active=7763. byte-identical. |
| output/backups/rentals_20260615_154400.db.gz | 10,048 | 11,165 | → 2026-01-16 | identical | active=7763. byte-identical. |
| output/backups/rentals_20260615_154813.db.gz | 10,048 | 11,165 | → 2026-01-16 | identical | active=7763. byte-identical. |
| output/backups/rentals_20260615_154814.db.gz | 10,048 | 11,165 | → 2026-01-16 | identical | active=7763. byte-identical. |
| output/backups/rentals_20260615_160656.db.gz | 10,048 | 11,165 | → 2026-01-16 | identical | active=7763. byte-identical (sha256 confirmed). |
| output/rentals_backup_20251206_073739.db | 2,174 | (none) | 2025-12-05 → 2025-12-06 | kf 719, ch 648, rm 585, fx 222 | Earliest snapshot. No price_history table yet. scraped_at only. |
| output/rentals_backup_20251206_153219.db | 4,578 | (none) | 2025-12-06 (intraday) | rm 2092, sv 1231, kf 556, ch 451, fx 218, jdw 30 | Dec-6 PM. No price_history table. Note sv 1231 (more savills than later). |
| output/rentals_backup_before_short_let_removal.db | 2,413 | 5,031 | 2025-12-06 → 2025-12-08 | rm 1071, kf 558, sv 339, ch 295, fx 127, jdw 23 | active=2413. Pre-short-let-removal Dec-8 state. |
| output/_validate_chestertons.db | 25 | 25 | 2026-06-15 | chestertons 25 | Scraper-validation scratch DB (#3-5). Tiny, current-date, not historical. |
| output/_validate_knightfrank.db | 50 | 50 | 2026-06-15 | knightfrank 50 | Validation scratch. |
| output/_validate_foxtons/rentals.db | 40 | 40 | 2026-06-15 | foxtons 40 | Validation scratch. |
| output/_validate_openrent/rentals.db | 28→**165** | 165 | 2026-06-15 → 2026-06-16 | openrent 165 | Validation scratch, **LIVE-CHANGING**: 28 rows at census time (10:25), 165 by 11:02 — a scraper agent is actively writing it. **Only store with `openrent` source.** |
| **ROOT /Users/.../rentnegotiation/rentals.db** | — | — | — | — | **NOW 0-BYTE** (mtime 2026-06-16 10:25). Task expected the 86KB `properties`-schema decoy — that decoy is the ARCHIVED one (next row). The root file was emptied/recreated recently (today). ⚠️ flag below. |
| scrapy_project/london_rentals.db (archived) | — | — | — | — | 0-byte stub (already archived in cleanup #1). |
| archive/_cleanup_2026-06-15/prototype_pre_scrapy/rentals.db | 0 (no `listings`) | 1 | — | — | The real **LEGACY DECOY** (86KB): `properties/agents/rooms/...` schema. properties=1 row. Not the live schema. DO NOT USE (confirmed decoy by dataeng). |
| output/rentals_modeler_copy.db | — | — | — | — | **0-byte** (mtime 2026-06-15 15:26). Empty placeholder; holds nothing. |

(rm=rightmove, ch=chestertons, kf=knightfrank, fx=foxtons, sv=savills, jdw=johndwood)

## JSONL

- **82 `*.jsonl` files** in `output/`, ~95,392 total lines (per-area scrape dumps, e.g. `belgravia_savills_listings.jsonl`). These are the raw scrape outputs that fed the DB rows; they do **not** extend the date range (no records beyond the Dec–Jan window) and are pre-dedup. Potential value: per-area raw records for cross-checking dedup losses, but they are a SUBSET timeline of what's already in the DBs, not a richer/newer source.

## Schema (canonical `listings` — 56 cols)

`id, source, property_id, url, area, price, price_pw, price_pcm, price_period, address, postcode, latitude, longitude, bedrooms, bathrooms, property_type, size_sqft, furnished, let_agreed, agent_name, agent_phone, summary, description, features, added_date, scraped_at, property_type_std, let_type, postcode_normalized, postcode_inferred, agent_brand, reception_rooms, size_sqm, epc_rating, floorplan_url, room_details, has_basement … floor_count, property_levels, canonical_id, address_fingerprint, first_seen, last_seen, is_active, price_change_count, is_short_let`
Tables: `listings, price_history, scrape_runs, scrape_events, listings_deduped, dedupe_stats, duplicate_groups` (+ `area_stats_clean, listings_clean` in canonical).
`price_history(id, listing_id, old_price, new_price, recorded_at, …)`.

## Answers to the task's two questions

1. **Most complete local dataset:** canonical `output/rentals.db` (= the 8 `.gz` + `pre_dataeng.bak`), all 10,048 rows + 11,165 price_history. ~~Earlier I called `pre_dataeng.bak` "most complete" for its 8,885 active count~~ — **corrected (dataeng): that 1,122-active surplus is the deliberately-deactivated stale set, not recoverable data.** Use canonical as the row/history base; **recompute `is_active` cycle-relative in the union (#36), do NOT copy it from the .bak.**
2. **Any local backup predating Jan-16 OR holding data we think is gone?** — **NO, on both counts.**
   - Predating: the Dec snapshots are *older* but *smaller* (fewer rows) — they hold nothing the canonical lacks. No local store reaches past Jan-16.
   - "Data we think is gone": the 1,122 active→inactive rows are correctly-inactive stale listings (NOT gone data). The only genuinely-not-in-canonical thing locally is the **`openrent` source** (165 rows in the validate scratch DB) — and the **lead ruled openrent is intentionally NOT canonical** (landlord-direct, different market segment; needs distribution analysis before model inclusion). The bulk gap (prod 22,317 vs local 10,048 ≈ 12,269 rows) is **NOT present locally** — it must come from Neon PITR (#31) / git archaeology (#34) / model-artifact reconstruction (#35).

## ⚠️ Flags

- **ROOT `rentals.db` is 0-byte as of today (2026-06-16 10:25)** — not the 86KB decoy the task referenced. Someone (a recovery agent?) touched/emptied it today. The real 86KB `properties`-schema decoy is safely archived at `archive/.../prototype_pre_scrapy/rentals.db`. Worth confirming nobody expected real data in the root file.
- **`rentals_modeler_copy.db` is 0-byte** — if any process expected modeler's copy to hold data, it's empty.
- **`damaged_allinactive_*.bak` (active=0)** is a poisoned snapshot — ensure no recovery step reads it by mistake.
- **`openrent`** exists ONLY in `_validate_openrent/rentals.db` (165 rows as of 11:02, was 28 at census 10:25 — live-changing scratch DB). **Lead ruled it is intentionally NOT canonical** (landlord-direct, different market segment; contract §4.1 — needs distribution analysis before model inclusion). So it stays scratch, NOT folded into the union. Logged for #39.
- **Validate scratch DBs are LIVE-CHANGING during this recovery window** (scraper agents writing them; max `last_seen` seen up to 2026-06-16 10:54+). The historical full DBs/backups are stable; only the `_validate_*` scratch DBs move. This census's full-DB numbers are stable; treat `_validate_*` counts as point-in-time.
