# SERVING FIX REPORT — Task #8

**Owner:** `serving` · **Status:** COMPLETE (prod write GATED on lead) · **Date:** 2026-06-15
**Depends on:** Data Layer Contract (`DATA_LAYER_CONTRACT.md`, dataeng #2)

Makes the deployed dashboard serve real data: a gated prod-sync from the canonical
SQLite DB → Neon Postgres, a fix for the empty-DB 500, and verified real comps/peers.

---

## TL;DR for the lead

1. **Prod write is GATED.** Nothing was written to prod. Run the load yourself
   **after rotating the `POSTGRES_URL` secret** (security #10). Exact command in §4.
2. **Empty-DB 500 is fixed** — the three `/report/*` pages now render a graceful
   "no data yet" page instead of crashing. Verified against a real empty Postgres.
3. **Sync is proven end-to-end** against an ephemeral real Postgres: 10,048 listings
   load, and the dashboard's own query SQL then returns **168 comparables**,
   **5,329 market listings**, and **61 peers** for the SW1W subject (was 0/0/0).

---

## 1. What was broken (health-team findings, root-caused)

| Symptom | Real root cause |
|---|---|
| `/api/similar` returns `peer_count:0` everywhere | Prod Neon `listings` table is EMPTY — no data was ever synced from the canonical SQLite DB. |
| `/report/negotiation` hard-500s on empty DB | NOT a missing error boundary (one exists). The 500 is **divide-by-zero / empty-array math AFTER the try/catch**: `getMedian([])`, `getPercentileValue([])`, `getPercentile([])`, `Math.round(lowerCount/comparables.length*100)` → `NaN`/`undefined`; and `getMarketStats()` returns one all-NULL row on an empty DB → `.toLocaleString()` on `null` throws. |
| `/report` + `/report/landlord-price` "only stay 200 via hardcoded constants" | Partially a mischaracterization: market values (comps, medians, percentiles, distribution) are already DB-driven via `getComparables`/`getMarketStats`/`getPpsfDistribution`. The only constants are the **subject-property inputs** (`4 South Eaton Place`: address, size, beds, landlord asking price) and the model fallback (used only if the `property_valuations` DB row is absent). Those are legitimate inputs, not fake market data. They stay. |

---

## 2. Changes made (files I own)

| File | Change |
|---|---|
| `scripts/sync_sqlite_to_postgres.py` | **NEW.** Prod-sync: canonical `output/rentals.db` → Neon Postgres. DRY-RUN by default; `--execute` gated behind `--i-have-rotated-the-secret`. |
| `scripts/_verify_serving_sync.py` | **NEW.** End-to-end verification harness (ephemeral real Postgres, runs the real sync + the dashboard's real query SQL). Dev/CI tool. |
| `dashboard/src/lib/db.ts` | `getMarketStats()` now coalesces the all-NULL empty-DB row to zeros, so callers never see `null`/`NaN`. |
| `dashboard/src/app/report/page.tsx` | Empty-state guard + `NoDataYet` panel. |
| `dashboard/src/app/report/negotiation/page.tsx` | Empty-state guard + `NoDataYet` panel (this is the page that 500'd). |
| `dashboard/src/app/report/landlord-price/page.tsx` | Empty-state guard + `NoDataYet` panel. |

**Empty-state guard** (each report page, right after the existing error boundary):
```ts
if (comparables.length === 0 && marketStats.total_listings === 0) {
  return <NoDataYet ... />;   // graceful page; returns BEFORE any NaN math
}
```

I did **not** edit `dashboard/src/app/api/init-db/route.ts`, `dashboard/next.config.js`,
or `dashboard/package.json` (security #10 owns those). I also did not edit
`api/similar/route.ts` or `api/running/route.ts` — both already degrade gracefully
(`similar` returns a 500 JSON on error / `peer_count:0` on no rows; `running` returns `[]`).

**Schema drift (Contract §3):** the parity columns `listings.canonical_id` and
`price_history.price_pw` are handled two ways: (a) security already added them to
`init-db/route.ts` (the authoritative schema creator), and (b) my sync script also
runs `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for both as an idempotent safety net,
so a load never fails on a column the canonical DB has but Postgres lacks.

---

## 3. Verification (REAL Postgres, no mocks)

`python3 scripts/_verify_serving_sync.py` spins up an ephemeral Postgres 16, applies
the `init-db` schema (intentionally WITHOUT `canonical_id`, to prove the sync's ALTER
adds it), runs the real sync with `--execute`, then runs the dashboard's exact query SQL.

**Loaded-DB results (PASS):**
```
listings_loaded:          10048
active:                   7763
price_history:            11165
canonical_id_populated:   175      (parity column added by sync + populated)
comparables_report:       168      (getComparables for the SW1 2-bed subject)
market_total:             5329     (getMarketStats)
market_median_ppsf:       5.51
similar_peers_SW1W:       61       (getSimilarListings — /api/similar)
```

**Empty-DB results (PASS):** `getMarketStats` → `total_listings:0`, comparables → 0,
`NoDataYet` guard fires → graceful page, **no NaN, no 500.**

Dashboard `npx tsc --noEmit` → exit 0. `npx next lint` → no errors/warnings in changed files.

---

## 4. GATED prod load — command for the LEAD to run

**Pre-req: rotate the `POSTGRES_URL` secret first (security #10), then:**

```bash
cd scrapy_project

# 1. Ensure the prod schema exists (security-owned init-db route).
#    Either hit the deployed route once:  curl https://<app>.vercel.app/api/init-db
#    (or run it locally against prod POSTGRES_URL per security's instructions).

# 2. DRY-RUN against prod (no writes — prints row-count deltas, then rolls back):
POSTGRES_URL='<ROTATED_PROD_URL>' python3 scripts/sync_sqlite_to_postgres.py

# 3. REAL load (only after the dry-run looks right):
POSTGRES_URL='<ROTATED_PROD_URL>' \
  python3 scripts/sync_sqlite_to_postgres.py --execute --i-have-rotated-the-secret
```

Expected after step 3: `listings ≈ 10,048`, `price_history ≈ 11,165`, `scrape_runs = 13`,
and `/api/similar` / `/report/*` immediately serve real data. The load is atomic
(`TRUNCATE ... + INSERT` in one transaction) and idempotent (safe to re-run).

> The sync **refuses** to write without BOTH `--execute` and `--i-have-rotated-the-secret`,
> so it cannot accidentally hit prod. The Vercel deploy remains gated on you.

---

## 5. Follow-ups / handoffs

- **Automation (#9):** this sync should run at the end of every scrape/retrain cycle so
  prod stays fresh. The script is import/CLI-friendly for the pipeline to call.
- **`scrape_runs` freshness:** synced telemetry is a frozen snapshot (last real scrape
  2026-01-16). The dashboard "running spiders" view will show none running — correct.
- **`postcode_inferred` type** (Contract §3 item 3, cosmetic INTEGER vs TEXT): not blocking;
  left to security/init-db owner.
