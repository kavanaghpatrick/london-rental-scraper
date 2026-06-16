# MERGE_PLAN — non-destructive UNION of all rental data (lose nothing)

**Task #36 (dataeng).** Date: 2026-06-16. Goal: produce ONE complete canonical dataset that is the UNION of every data source, losing zero unique listings and zero price-history rows. **Non-destructive**: never overwrite newer data, never drop a row that exists in only one source.

## Inputs to merge (data inventory)

| Source | Listings | price_history | Notes |
|---|---|---|---|
| **Local canonical** `output/rentals.db` | 10,048 | 11,165 | Frozen at 2026-01-16. The richest per-row data (full fields, fingerprints, floors). |
| **Prod-recovered** (Neon PITR → `prod_recovered.db`/branch, #31) | ~22,317 | ~30,145 | The big one. Jan→May. ~12,269 listing IDs + ~18,980 history rows NOT in local (per #32). PENDING #31. |
| **Artifact extract** `output/prod_only_ids_from_2026-05-21_artifact.json` (#32/#35) | 1,475 IDs | — | DERIVED fields only (fv/lo/hi/pct/sq). LAST-RESORT fill, not a primary source. |
| **JSONL / git-history extracts** (#33/#34) | TBD | TBD | Fold in only if they contain IDs absent from both above. |

> The merge is **gated on #31 (PITR)**. Prod-recovered is the dominant source; do not run the merge until it lands (or PITR is declared impossible — then artifact extract becomes the only prod fill, with the ~88% data-loss caveat flagged to lead).

## Identity & dedup keys (reuse existing logic — do NOT invent new)

1. **Primary identity = `(source, property_id)`** — the `UNIQUE` constraint on `listings`. This is the merge join key. Two rows with the same `(source, property_id)` are the SAME listing.
2. **Cross/same-source dup = `address_fingerprint`** — the 16-char hash from `property_scraper/services/fingerprint.py` (already on all local rows; regenerate for any prod row missing it via the SAME function — identical input → identical hash, deterministic). Used only for the *optional* post-merge dedup pass (`dedupe_cross_source.py`), NOT for the union itself. **The union keys on `(source, property_id)`; fingerprint dedup is a separate, logged, non-deleting step.**
3. **price_history identity = `(source, property_id, price_pcm, recorded_at)`** AFTER remapping (see ⚠️ below). Locally `(listing_id, price_pcm, recorded_at)` is 100% unique (11,165/11,165), so this triple is a safe dedup key.

## ⚠️ THE critical correctness rule: never merge `price_history` by `listing_id`

`price_history.listing_id` is a FK to `listings.id`, an **autoincrement surrogate** (local max = 31,054). **`id` values are NOT comparable across databases** — local id=5 and prod id=5 are different listings. Merging history by raw `listing_id` would silently attach prod price history to the wrong local listings = data corruption.

**Rule:** resolve every `price_history` row to its listing's `(source, property_id)` in its OWN database first, then re-attach to the unified listing's NEW id. Concretely: `price_history → JOIN its own listings ON listing_id → carry (source, property_id, price_pcm, price_pw, recorded_at)` as the portable record; insert into merged DB looking up the new `listing_id` by `(source, property_id)`.

## Merge procedure (step-by-step, non-destructive)

**Base DB choice:** build a NEW `output/rentals_merged.db` (never mutate local canonical or the recovered DB in place). Start it as an exact copy of **prod-recovered** (it's the superset — ~22,317 ⊇ most of local). Then UPSERT local on top to recover local-only IDs and local's richer fields. Rationale: local is frozen at Jan-16 with the best per-row completeness; prod has the breadth. We want breadth (prod) + depth (local) with no loss.

### Step 0 — Pre-flight (safety)
- Back up every input (`.bak` copies). Record row counts of each. Abort if any source unreadable.
- Verify schema parity between local and recovered `listings` (54 cols per DATA_LAYER_CONTRACT §2); ALTER recovered to add any missing cols (e.g. `canonical_id`) before merge so columns align.

### Step 1 — Seed merged DB from the broader source
- `rentals_merged.db` = copy of `prod_recovered.db` (≈22,317 listings, ≈30,145 history). If PITR failed, seed from local instead and treat artifact extract as the only prod fill.

### Step 2 — UPSERT local listings into merged (key = `source, property_id`)
For each local listing:
- **If `(source, property_id)` NOT in merged** → INSERT (this is a local-only listing; lose nothing).
- **If it EXISTS in merged** → UPDATE **field-by-field with COALESCE-style precedence**, NOT a blind overwrite:
  - **Temporal:** `first_seen = MIN(both)`, `last_seen = MAX(both)`. Keep the EARLIEST first_seen and LATEST last_seen across sources.
  - **`is_active`:** OR-of-truth is wrong here — recompute cycle-relative AFTER merge (see Step 5). Do not copy is_active from either side blind.
  - **Structural/enriched fields** (`size_sqft, size_sqm, bedrooms, bathrooms, address, postcode, lat/long, floorplan_url, floor flags, epc_rating, agent_*`): prefer the NON-NULL value; if both non-null, prefer the row with the LATER `last_seen` (most recent observation), EXCEPT keep a non-null over a null even if older (don't let a newer null erase an older value — same COALESCE-sticky rule the pipeline uses).
  - **`price_pcm/price_pw/price`:** take from the LATER `last_seen` row (most recent price), but log the older price into price_history if not already present (Step 3 captures this).
  - **`address_fingerprint`:** keep existing; if missing on the merged row, regenerate via fingerprint.py.
  - `price_change_count`: recompute from the unioned price_history (Step 3), don't sum.
- **Never** delete a merged row because it's absent from local. Local is a frozen subset; prod-only rows stay.

### Step 3 — UNION price_history (remap by source:property_id, dedup by the triple)
- From EACH source DB, extract portable history: `SELECT l.source, l.property_id, ph.price_pcm, ph.price_pw, ph.recorded_at FROM price_history ph JOIN listings l ON ph.listing_id = l.id`.
- Insert into merged `price_history`, resolving `listing_id` via `(source, property_id)` lookup in the merged listings.
- **Dedup key:** `(source, property_id, price_pcm, recorded_at)` — skip rows already present. This unions both histories losing nothing and double-counting nothing.
- Recompute each listing's `price_change_count` = distinct price points in its merged history − 1 (floor 0).

### Step 4 — (Optional, logged) fingerprint dedup pass — NEVER auto-delete
- Run `dedupe_cross_source.py --analyze` against merged to REPORT cross-source duplicates (same property on Rightmove + agent site). Mark with `canonical_id`; do NOT `--remove`. Deletion, if ever wanted, is a separate lead-approved step with a logged manifest. The UNION's job is to PRESERVE; dedup is advisory.

### Step 5 — Recompute `is_active` cycle-relative (per DATA_LAYER_CONTRACT §5.1)
- After merge, `MAX(last_seen)` advances to prod's latest (~2026-05-14). Apply the §5.1 rule: `is_active = (last_seen >= MAX(last_seen) − 7d)`. This yields a correct active set for the MERGED recency, not the stale Jan-16 anchor. (Wall-clock still avoided — the merged snapshot is still frozen until a fresh scrape.)

### Step 6 — Verify (acceptance criteria)
- `merged.listings >= MAX(local, prod_recovered)` and `>= |union of all (source,property_id)|`. Specifically expect **≈22,317 + local-only-not-in-prod** (likely ~0 extra if prod is a true superset of local IDs, but verify — any local ID absent from prod MUST appear in merged).
- `merged.price_history >= MAX(local_ph, prod_ph)` and equals `|union of (source,property_id,price_pcm,recorded_at)|`.
- 0 orphan price_history (every `listing_id` resolves).
- No `(source, property_id)` from ANY input is missing from merged (assert via set-difference).
- integrity_check OK; per-source counts logged before/after.

## Conflict-resolution summary (one-line rules)

| Field | Rule |
|---|---|
| `first_seen` | MIN across sources |
| `last_seen` | MAX across sources |
| structural/enriched | non-null wins; if both non-null, later `last_seen` wins (sticky: never null-over-value) |
| `price_*` | from later `last_seen`; older prices preserved in price_history |
| `is_active` | recomputed cycle-relative post-merge (§5.1) |
| `price_history` | UNION, dedup by `(source,property_id,price_pcm,recorded_at)`, remap by source:property_id NOT id |
| duplicates (fingerprint) | REPORT only (canonical_id), never auto-delete |

## Script outline (`merge_datasets.py`, dataeng-owned)

```
def merge(prod_db, local_db, out_db='output/rentals_merged.db', artifact_json=None, dry_run=True):
    preflight_backup_and_counts([prod_db, local_db])
    align_schema(prod_db, local_db)                 # add missing cols both ways
    seed = prod_db if exists(prod_db) else local_db  # Step 1
    copy(seed -> out_db)
    others = [local_db] (+ [prod_db] if seed==local_db)
    for db in others:                                # Step 2
        for row in listings(db):
            upsert_listing(out, row)                 # key=(source,property_id), COALESCE/last_seen precedence
    union_price_history(out, [prod_db, local_db])    # Step 3: remap by source:property_id, dedup triple
    if artifact_json: fill_from_artifact(out, artifact_json)  # last-resort, derived-only, flagged
    mark_inactive_cycle_relative(out)                # Step 5 (§5.1 rule)
    report_fingerprint_dupes(out)                    # Step 4, analyze-only
    verify(out)                                      # Step 6 assertions; raise on any loss
    if not dry_run: promote(out -> output/rentals.db)  # only after lead sign-off + verify pass
```

- **Default `dry_run=True`** — prints the merge diff (rows added per source, history unioned, conflicts) WITHOUT writing canonical. Promote to `output/rentals.db` only after verify passes AND lead sign-off.
- Reuses: `property_scraper/services/fingerprint.py` (fingerprints), `dedupe_cross_source.py` (analyze), DATA_LAYER_CONTRACT §5.1 (is_active).
- dataeng is sole writer; merge runs against COPIES, promotes once.

## Status — script BUILT and dry-run VERIFIED on available inputs (2026-06-16)

`merge_datasets.py` is implemented and ran clean as a dry-run on what we have NOW (base + git-recovery), structured so the PITR dump folds in via `--prod-db`:

- **Dry-run result:** local 10,048 + git-recovery 804 (all NEW, 0 already in canonical, full 54-col records, Dec 6–15 tail) = **10,852 listings**, 0 orphan history, integrity OK, is_active recomputed cycle-relative (7,763 active — the 804 Dec records correctly land inactive but are PRESERVED with full data).
- Per-source added: rightmove +713, chestertons +51, knightfrank +32, savills +5, foxtons +3.
- **Re-scrape seed generated** (NOT inserted, per lead): `output/rescrape_seed_urls.txt` = 1,379 rightmove URLs (predictions-only IDs lacking raw fields). 96 non-rightmove prod-only IDs flagged to scraper-rm-sav for source-specific URLs.

### ⛔ Neon PITR recovery is DEAD — no prod dump is coming (confirmed by lead 2026-06-16)
The user checked the Neon console: **Free tier, 6-hour PITR retention, truncate was ~18h ago → no restore point exists.** The ~22,317 pre-truncate prod rows (the ~12,269 prod-only Feb–May listings) are **NOT recoverable**. The `--prod-db` slot in `merge_datasets.py` will simply go unused. The COMPLETE recoverable dataset is therefore: **post-scrape `output/rentals.db` (local Jan-16 + today's fresh re-scrape inventory) + the 804 git-recovered full records + the 1,379-URL rightmove re-scrape seed**. Anything beyond that is only re-obtainable by re-scraping live sites (the seed list), not from any backup.

### is_active: cycle-relative ONLY (lead CONFIRMED — do NOT pull from the .bak)
Lead confirmed 2026-06-16: DROP the ".bak pull is_active" step entirely. After the fresh scrape, `MAX(last_seen)` advances to today, so cycle-relative recompute (§5.1) against today's anchor is correct and well-defined — anything not re-seen in today's scrape (incl. the Dec/Jan stale rows AND the 804 Dec git records) lands inactive, which is right. The pre_dataeng `.bak` holds those 1,122 stale rows as ACTIVE (pre-mark-inactive state); pulling from it would wrongly re-activate them. **Cycle-relative only.**

## Dependencies / status

- **UNBLOCKED.** No external data dependency remains — PITR is dead, so there is no prod dump to wait for. Base + git-recovery merge is real and dry-run-verified NOW.
- **Only remaining gate:** the live recovery scrape (#41) must FINISH before I run the merge against canonical (can't read/merge a DB an active scraper is writing). Waiter armed.
- Execution: scrape finishes → `merge_datasets.py` dry-run (post-scrape rentals.db + 804 git records) → cycle-relative is_active → verify → backup (golden-master + .bak) → `--promote` → ping modeler (#43).
- Pairs with **#37** (DONE — prod-sync is now non-destructive UPSERT) so the merged dataset syncs to prod without re-destroying it.
- End state: the COMPLETE recoverable dataset that modeler retrains v20 on (#43).
