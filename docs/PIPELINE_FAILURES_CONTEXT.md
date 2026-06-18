# Pipeline Failures — Shared Context (2026-06-17)

The automated daily pipeline did NOT run end-to-end today. Three distinct, confirmed
failures. This doc is the team's shared starting evidence — do deeper root-cause from here.

## Methodology: TDD (mandatory)
For every fix: (1) write a test that REPRODUCES the bug and FAILS (Red), (2) confirm it
fails for the right reason, (3) implement the minimal fix (Green), (4) run the full suite
for regressions, (5) refactor if needed. No production fix lands without a failing-first test.

Run the suite with: `python3 -m pytest tests/ -q`
CI-equivalent gates: pytest+coverage, `node chrome-extension/_fixture_diff.mjs`, `tsc --noEmit` (dashboard).

---

## PROBLEM 1 — Dedupe deletes 39.5% of active listings (BLOCKS the end-to-end pipeline)
**Evidence** (Daily Property Scrape run 27673895986, 2026-06-17 07:47Z):
```
✓ Run scraper (Postgres mode)
✓ Run OCR enrichment / Mark stale inactive
✗ Clean duplicate listings (address similarity)   <-- FAILED, exit 1
- Retrain CANONICAL model (v20)                    <-- SKIPPED (so model never retrained today)
- Commit canonical model files                     <-- SKIPPED
```
Step error:
```
Dedupe DELETE aborted: would remove 3095/7831 (39.5%) active listings —
exceeds 10% safety threshold. Likely a clustering bug. NOT deleting.
```
The safe-delete guard (`scripts/_safe_delete.py`) correctly refused — but exit 1 skips retrain.
**Open question to research:** is 39.5% an over-clustering BUG (address-similarity matching
distinct properties on the same street) or LEGITIMATE cross-source duplication (rightmove
re-listing what agents post directly)? Decide fix vs threshold accordingly — do NOT just bump
the threshold without proving the clusters are correct.
**Files:** `dedupe_cross_source.py`, `scripts/dedupe_same_source.py`, `scripts/_safe_delete.py`,
the "Clean duplicate listings" step in `.github/workflows/daily-scrape.yml`. Runs in Postgres mode.

## PROBLEM 2 — Neon backup fails daily (pg_dump version) + same in any CI prod path
**Evidence** (Neon Prod Backup run 27671674788, 2026-06-17 07:01Z):
```
pg_dump: error: aborting because of server version mismatch
server version: 17.10 ; pg_dump version: 16.14 (Ubuntu 16.14-1.pgdg24.04+1)
[backup_neon] ERROR: pg_dump failed — NO backup written.
```
CI installs postgresql-client **16**; Neon server is **17**. Same root cause hit locally
(local needs `/opt/homebrew/opt/postgresql@17/bin`). Affects `neon-backup.yml` and any workflow
that calls `scripts/backup_neon.sh` before a prod write.
**Also note:** Node20-action deprecation warnings across workflows (checkout@v4, setup-python@v5,
upload-artifact@v4, github-script@v7) — in scope to assess (not necessarily fix now).
**Files:** `.github/workflows/neon-backup.yml`, `daily-scrape.yml`, `scripts/backup_neon.sh`.

## PROBLEM 3 — CI unit test fails: unseen-postcode default freq (np.log1p TypeError)
**Evidence** (CI run 27636202638 on push 2bbbf12):
```
FAILED tests/test_model_inference.py::test_unseen_postcode_uses_default_freq
TypeError: loop of ufunc does not support argument 0 of type float which has no callable log1p method
```
Regression from the artifact regen: `output/rental_model_canonical_inference.json` now stores
the in-dict `postcode_freq['default']` as `None` (real default moved to a separate
`postcode_freq_default` key). An unseen postcode → freq=None → object-dtype column →
`np.log1p` blows up. The read path likely uses `postcode_freq.get(pc, postcode_freq['default'])`.
**Files:** `gen_inference_stats.py` (writes inference.json), `canonical_predict.py` +
`scripts/_canonical_features.py` (read path), `output/rental_model_canonical_inference.json`,
`tests/test_model_inference.py::test_unseen_postcode_uses_default_freq`.
**Watch:** keep JS↔Python parity green (`_fixture_diff.mjs`) — the JS freq-default must match
whatever Python does. Don't reintroduce the social-housing target leak (see memory).

---

## Guardrails (do NOT regress)
- Non-destructive prod sync (UPSERT), safe-delete guard, the leak-free model (R²=0.792).
- Live now: /api/predict, /api/similar, GitHub-raw artifacts — keep them working.
- All canonical data is precious; never bypass `_safe_delete.py`.
