# TRUNCATE IMPACT ANALYSIS — was prod's 22,317 real data?

**Task #32 (dataeng).** Date: 2026-06-16. Question: did today's prod TRUNCATE+reload destroy real scraped data we don't have locally, or just duplicate/stale bloat?

## VERDICT

**We lost approximately 12,269 real unique listings and ~18,980 price-history rows, spanning ~2026-01-16 → 2026-05-14 (prod's last scrape cycle). These were genuine, NOT bloat.**

**⛔ UPDATE 2026-06-16 — Neon PITR recovery is CONFIRMED DEAD (NOT recoverable).** The user checked the Neon console: Free tier = 6-hour PITR retention; the truncate was ~18h ago; no snapshots exist → there is NO pre-truncate restore point. The full prod dataset cannot be restored. What IS recoverable: (a) the 804 git-recovered full records (Dec tail), (b) today's fresh re-scrape inventory (#41), (c) ~1,379 still-live rightmove listings via the re-scrape seed (`output/rescrape_seed_urls.txt`) — re-obtained from live sites, not from any backup. The ~12,269 prod-only Feb–May listings are otherwise permanently lost. This is why the data-safety regime (#37 non-destructive sync, #38 backups, #46 tests) matters: there was no second copy.

## The numbers

| | Local canonical | Prod (pre-truncate) | Delta (lost) |
|---|---|---|---|
| listings | 10,048 | 22,317 | **12,269** |
| price_history | 11,165 | 30,145 | **18,980** |
| scrape_runs through | 2026-01-16 | 2026-05-14 | ~4 months |

## Why the 12,269 are REAL unique listings, not bloat (3 independent proofs)

1. **Schema makes dup bloat impossible.** Prod `listings` has `UNIQUE(source, property_id)` (confirmed in `dashboard/src/app/api/init-db/route.ts` and `pipelines_postgres.py`). 22,317 rows therefore = **22,317 DISTINCT `(source, property_id)` pairs**. You cannot accumulate duplicate listings under that constraint.

2. **Prod's writer was an UPSERT, not an append.** Prod was populated by `PostgresPipeline` (smart upsert: INSERT new `(source,property_id)`, UPDATE existing in place). Row count only grows when a genuinely NEW listing ID appears. Re-scraping the same listing UPDATEs, never duplicates. So every one of the 12,269 extra rows is a listing ID that was never in our Jan-16 snapshot.

3. **The accumulation rate is normal London churn.** 12,269 new IDs over the ~118-day Jan16→May14 window = **~104 new distinct listings/day across 5 sources**. London has tens of thousands of active rentals (Rightmove alone lists thousands); ~104 new IDs/day from listing turnover (lets, withdrawals, new instructions) is low-to-normal, not anomalous. Daily scrapes confirmed running: **34 daily V20 model auto-commits Apr16→May21** in git history.

**price_history corroborates:** 18,980 extra history rows / 12,269 extra listings = **1.55 rows per new listing** — exactly what real data looks like (every listing logs an initial price; some log subsequent changes). Bloat would not produce this clean ratio, and `price_history.listing_id` is FK-bound to real listings.

## Estimated loss per source (delta apportioned by local source distribution)

| source | est. prod-only listings lost |
|---|---|
| rightmove | ~8,478 |
| chestertons | ~1,357 |
| knightfrank | ~1,103 |
| foxtons | ~670 |
| savills | ~660 |
| **total** | **~12,269** |

(Estimate; exact per-source split is only knowable from the recovered prod dump — see caveat.)

## chrome-extension artifacts as a partial backup (nuanced — version matters)

There are TWO different versions of these artifacts, and the distinction matters:

- **Working-tree / 2026-06-15 rebuild version** (`predictions.json` 7,048 keys, `similar_listings.json` 7,741): regenerated from LOCAL canonical (the `similar_listings.json` 7,741 has exactly ONE commit in history — `6edf08b`, 2026-06-15, a local rebuild; absent at every prod-era commit). All 9,478 distinct keys are a 100% subset of local 10,048 → **0 prod-only IDs**. Useless for recovery; usable only as a FIELD-BACKFILL source (lat/lon/baths for rows already in canonical), never as new-row provenance.
- **The `predictions.json` on origin/main** — last MODIFIED 2026-01-16 (commit `9dfd963`), still present unchanged at the final prod commit 2026-05-21 (`350856d`), i.e. a **JANUARY snapshot, NOT Apr–May data** (the daily Apr16–May21 commits only re-wrote `model.json`; predictions.json was never re-committed, so `350856d` and `9dfd963` reference the same blob `9f760443`). It has **9,159 keys, of which 1,475 (16.1%) are prod-only IDs NOT in local** — by source: rightmove 1,379, knightfrank 30, chestertons 29, savills 24, foxtons 13. A genuine partial backup, but a **January** one — the Apr–May bulk was never in any artifact.

**But it's a THIN partial backup:** it recovers only **~1,475 of the ~12,269 lost IDs (~12%)**, and only DERIVED fields per listing: `{fv (predicted pcm), lo, hi, pct (asking-vs-predicted %), sq (sqft)}`. **No address, bedrooms, raw asking price, postcode, or agent.** You can back out an *approximate* asking price from `fv × (1 + pct/100)`, but it's lossy and has no structural/location fields. `similar_listings.json` is **0 bytes at every prod-era commit** (the empty-export symptom — it filtered on is_active which was already broken), so it contributes nothing.

**Conclusion:** artifacts are a ~12%, lossy, derived-only, JANUARY-only safety net — useful as a cross-check, but NOT a substitute for full data, and they hold nothing past January. With **Neon PITR confirmed DEAD** (see top), the full 12,269 lost listings are NOT recoverable from any backup; only the still-live ones are re-obtainable via re-scrape. (This bounds #35's premise: the artifact has 1,475 prod-only IDs, but without raw listing data and only from January.)

### Net git-recovery accounting (reconciled with scraper-rm-sav #34 + modeler #35)
- `output/rentals.db @ 92a83c9` (Dec-15) → **804 full records** (prod-era, real fields — merged).
- `predictions.json @ 9dfd963` (Jan-16) → 9,159 IDs; **1,475 prod-only**, of which **226 already covered by the 804** → **1,249 partial stubs** (ID + fv + sqft only) → re-scrape seed.
- `similar_listings.json` → LOCAL rebuild, NOT prod; 0 new rows; field-backfill only.
- **Net new prod listings from git = 2,053** (804 full + 1,249 partial). Merged dataset = post-scrape `rentals.db` + the 804 full records; the 1,249 partials route to the re-scrape seed (`output/rescrape_seed_urls.txt` 1,167 rightmove + `output/rescrape_seed_nonrightmove.csv` 82).

## Caveat on precision

The 12,269 / 18,980 figures are exact **deltas** between two known counts (local 10,048 / 11,165 vs prod 22,317 / 30,145 per `AUTOMATION_REPORT.md` line 170 + the lead's stated prod price_history count). The per-source split and the exact date histogram of the lost rows are **estimates** until the pre-truncate prod state is recovered via Neon PITR (#31). Some fraction of the 12,269 prod-only IDs may be listings that went inactive before May (still real data with valid price history — worth preserving for the model and for time-series).

## Recommendations

1. **Recover, don't recreate.** This data cannot be re-scraped (those listings are gone from the live sites). Neon PITR/branch-from-timestamp (#31, serving) is the only path. Prioritize it — window closes within Neon's retention (24h–7d from today).
2. **After recovery, UNION not overwrite.** Merge recovered prod (22,317) with local canonical (10,048) preserving every unique `(source, property_id)` and full price_history — don't let either side clobber the other (#36).
3. **Make prod-sync non-destructive.** The TRUNCATE+reload sync is what caused this. Switch to UPSERT (`ON CONFLICT (source, property_id) DO UPDATE`) so a sync can never again delete prod rows that aren't in local (#37).
