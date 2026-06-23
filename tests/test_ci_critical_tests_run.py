"""ANTI-SILENT-SKIP META-TEST (#49 systemic fix).

WHY THIS EXISTS
---------------
CI has repeatedly gone "green" while a regression test silently stopped protecting the
platform — because the test was renamed, de-collected, or skipped in the CI env. Two
real incidents motivated this: a test that imported PyYAML (absent in CI) and a test
that hard-asserted a local-only output/r3_baseline/ directory. Both turned a real test
into a no-op in CI, so "green locally" meant nothing.

This meta-test makes that failure class LOUD. It pins a small ALLOWLIST of CRITICAL
regression tests that MUST actually EXECUTE in PR CI (they have no real binary
dependency — no DB, no tesseract, no Postgres, no node), and FAILS if any of them is:
  * MISSING  — not collected at all (renamed / moved / deleted / typo'd marker), or
  * SKIPPED  — collected but skip-marked in the CI env (an over-broad skip that quietly
               disables a guard that should always run).

It runs inside the normal python-tests job, so a critical test silently going dark
fails the build instead of hiding behind a passing suite.

THE CONTRACT (contributor rule)
-------------------------------
A test on CRITICAL_TESTS is load-bearing: it must stay CI-safe (no DB/OCR/Postgres/node
deps) and must always run on every PR. If you must legitimately move/rename one, update
this allowlist in the SAME change. If a critical test genuinely needs a binary dep, it
does NOT belong on this list — that's the whole point (a test that can skip in CI gives
zero PR protection). Add new always-run guards here as they land.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Critical regression guards that MUST run on every PR (all CI-safe: no DB/OCR/PG/node).
# Use bare function/parametrized-base nodeids — parametrization is matched by prefix so
# we don't pin to a specific param id that may evolve.
CRITICAL_TESTS = [
    # --- Model / feature-engineering invariants (artifacts are git-committed → run in CI)
    "tests/test_model_inference.py::test_unseen_postcode_uses_default_freq",
    "tests/test_model_inference.py::test_social_housing_does_not_fire_priceless_prime",
    "tests/test_model_inference.py::test_coordless_row_distance_is_numeric_no_log1p_crash",
    # --- Spider selector-regression (site markup/JSON-shape drift)
    "tests/test_spider_parsing.py",
    # --- Scrapy version contract (the start()->start_requests() bridge). Scrapy 2.16.0
    #     removed that bridge, silently zeroing BOTH the for-sale + rental scrapes on
    #     2026-06-23 (every spider builds URLs in the legacy sync start_requests() and
    #     leaves start_urls empty). PR CI cannot run a real crawl, so this pure-unit guard
    #     (version ceiling + base-start bridge-source + spider start_requests non-empty) is
    #     the only PR-time net for a silent dependency upgrade. Must never go dark.
    "tests/test_scrapy_version_pin.py",
    # --- FOR-SALE vertical data layer (the separate sale-price tool's foundation):
    #     the for-sale parse seam + isolated sale_listings schema. CI-safe (reads the
    #     committed for-sale __NEXT_DATA__ fixture + in-memory sqlite; no DB/PG/OCR/node)
    #     so it ALWAYS runs and gates the PR. Includes the rental-isolation + no-pcm-leak
    #     guards — must never silently go dark, or a sale price could leak into the rent
    #     magnitude / the for-sale layer could couple to the parity-gated rental model.
    "tests/test_for_sale_data_layer.py",
    #     Companion: the for-sale PARSE-SEAM + isolated-item/validator contract
    #     (Rightmove+Foxtons __NEXT_DATA__ for-sale fixtures). Same CI-safety profile
    #     (committed fixtures, no DB/network); pins the sale-magnitude validator (rejects
    #     a rent leaking in) and the no-price_pcm-field isolation of the sale item.
    "tests/test_for_sale_scrape_layer.py",
    #     Companion: the BASELINE SALE-PRICE model (for_sale/sale_price_model.py) — the
    #     for-sale analogue of the rental v20 model, trained on the COMMITTED deterministic
    #     for-sale sample (no DB/network/node). Pins the leak-safety guard (asking_price is
    #     never a feature; no target-derived branch), the sale-magnitude/ppsf bounds, and
    #     the hard rental-isolation guard (must not import rental_price_models_v20 /
    #     canonical_predict). If this silently went dark a sale model could re-couple to the
    #     parity-gated rental chain or regress to a leaky/rent-magnitude baseline.
    "tests/test_for_sale_sale_model.py",
    #     Inc3: the FULL sale-price model (for_sale/sale_retrain.py + sale_predict.py +
    #     sale_features.py + the freq-map/inference plumbing in sale_price_model.py). The
    #     for-sale analogue of the rental v20 retrain/serving chain, trained on the
    #     COMMITTED deterministic sale sample (no DB/network/node), seed=42, artifacts to
    #     tmp_path. Pins the baked-freq inference fix (BLOCKER-1), the monotone size
    #     constraint, the artifact round-trip + Booster-JSON contract, determinism/seed
    #     stability, and the rental-isolation guard (now also banning retrain_canonical).
    #     If this went dark, the sale model could re-couple to the parity-gated rental
    #     chain or regress to the single-row freq degeneracy.
    "tests/test_for_sale_inc3_model.py",
    #     Inc4a: the for-sale SERVING contract (Python side). Pins the inference=True
    #     golden-writer fix (BLOCKER-1: district_freq != 1.0 via the baked maps), the
    #     gen_sale_golden.py scores-the-committed-Booster invariant (NOT a retrain), the
    #     'prediction_price' (not prediction_pcm) golden key, the inference.json
    #     default-shape (default == min(map), no nested 'default'), and the sale-artifact
    #     isolation (output/sale_api/, never chrome-extension/api/). CI-safe (committed
    #     fixtures + xgboost/numpy; no DB/PG/OCR/node). If this went dark the JS↔Python
    #     0/0 parity gate could silently drift or the serving golden re-degenerate.
    "tests/test_for_sale_inc4_serving.py",
    #     Inc2: the for-sale CLI + PIPELINE ROUTING contract. The --listing-type seam,
    #     get_sale_db_path() read seam, the SaleListingPipeline (writes output/sales.db),
    #     and the per-item discriminator guard that keeps sale items OUT of the rental
    #     listings table. CI-safe (CliRunner + in-memory/tmp sqlite; no DB/PG/OCR/node).
    #     Carries the CORE no-leak guard + the AMENDMENT-4 end-to-end regression; must
    #     never go dark, or a sale row could leak into rentals.db/listings.
    "tests/test_for_sale_cli_routing.py",
    #     Inc2: the three Playwright for-sale PARSE SEAMS (savills/knightfrank/chestertons
    #     parse_*_for_sale). Pins the sale-magnitude/no-price_pcm isolation, the MANDATORY
    #     Chestertons /sales/ stable-id fix, and the rental-model isolation guard. CI-safe
    #     (committed card-dict fixtures + in-memory parse; no DB/network/node).
    "tests/test_for_sale_playwright_parse.py",
    #     Inc2: the Playwright spider SALE-MODE WIRING (listing_type kwarg + coercion,
    #     the for-sale start-URL section seam, and the delegating parse_card_data_for_sale
    #     that routes to the pure seam with the 0->None boundary). Pins the zero-regression
    #     default-rental section. CI-safe (no crawl/network; committed fixtures).
    "tests/test_for_sale_playwright_spider_mode.py",
    #     Inc2 prod-scrape: the FOR-SALE GHA WORKFLOW meta-test (text-parse of
    #     .github/workflows/for-sale-scrape.yml — no DB/PG/node). Pins the prod-write
    #     SAFETY invariants of the scheduled sale scrape: scheduled+dispatch-ONLY triggers
    #     (NO push/pull_request, so merging never auto-fires a prod write), the real
    #     `sync_sales_to_postgres.py --execute --i-have-rotated-the-secret` invocation, NO
    #     destructive SQL (TRUNCATE/DROP/DELETE/setval) anywhere in the body, scrape-before-
    #     sync ordering, and the rental-isolation guard (no rentals.db/listings/retrain/
    #     sync_sqlite_to_postgres.py). If this went dark, an unsafe prod-write workflow (or a
    #     push-triggered auto-scrape of prod) could land unnoticed.
    "tests/test_for_sale_scrape_workflow.py",
    #     Inc2 prod-scrape: the IN-PROCESS sale-sync SAFETY layer (Layer-1; a SqlitePgCursor
    #     adapter runs the REAL load_sale_table/backup_prod against an in-memory SQLite
    #     stand-in prod — no PG binary/node). The always-green PR analogue of the real-PG
    #     job: prod-only row survives the UPSERT, first_seen-never/last_seen-updated, backup-
    #     before-write, >5% shrink delta-abort, idempotent re-run (zero dupes per
    #     source/property_id), conflict-key RuntimeError refusal. If this went dark, a
    #     destructive/clobbering sale-sync regression would be invisible in PR CI. (The
    #     real-Postgres deepening, tests/test_sale_sync_real_pg.py, is NOT on this allowlist
    #     because it skips without a PG container; it is pinned instead by the
    #     sale-sync-pg ci.yml job-existence meta-assert below.)
    "tests/test_sale_data_layer_safety.py",
    # --- Data-layer safety (the destructive-op guards / _safe_delete)
    "tests/test_safe_delete.py",
    "tests/test_data_layer_safety.py",
    # --- Cycle-relative mark-inactive (the recurring nightly prod is_active wipe).
    #     Pure file-parse + in-memory SQLite — no DB/PG/node. Pins the SHIPPED
    #     daily-scrape.yml SQL to MAX(last_seen), not wall-clock NOW(); must never
    #     silently stop running, or the wall-clock-wipe footgun could return unguarded.
    "tests/test_mark_inactive_cycle_relative.py",
    # --- Wave 3: the LIVE-SMOKE layer (scheduled extraction-drift + prod-schema-drift).
    #     Pins that both non-PR-blocking smoke workflows stay scheduled + issue-on-failure
    #     and the schema probe stays read-only. Pure text-parse (no DB/PG/node), so it
    #     always runs; if it went dark, the only net that catches live-site DOM migration
    #     or prod schema drift could be deleted silently.
    "tests/test_live_smoke_workflows.py",
    # --- Cross-source dedupe identity core (pure, no DB)
    "tests/test_dedupe_postgres.py::test_distinct_streets_not_merged",
    "tests/test_dedupe_postgres.py::test_true_cross_source_dupe_still_caught",
    # --- Wave 2 (A5/A12): the cross-source REMOVE path's cascade-delete +
    #     price_history-backup safety guard. Pins that dedupe_cross_source.remove_duplicates
    #     cascade-deletes the orphaned price_history rows AND routes the delete through the
    #     _safe_delete guard (delta-abort + backup), so the destructive remove path can't
    #     silently orphan history or bypass the recoverable-backup guard. Pure in-memory
    #     sqlite + CSV (no live DB/network/node), so it always runs in PR CI.
    "tests/test_dedupe_cross_source_remove.py",
    # --- M18: the dedupe identity PRIMITIVE itself (address fingerprinting). The whole
    #     cross-source identity rests on these — a regression in normalization/collision
    #     handling silently re-introduces the over-/under-merge bug. Pure-unit (no
    #     DB/network/node), so it ALWAYS runs and must never go dark behind MIN_EXECUTED.
    "tests/test_fingerprint.py",
    # --- M18: the SQLite smart-UPSERT contract (first_seen-once / last_seen-updated /
    #     price-change -> price_history). The relist + history-tracking core; pure-unit
    #     (in-memory sqlite via tmp_path, no live DB/network/node). If this stopped
    #     running, a broken upsert (dropped history, clobbered first_seen) would be
    #     invisible in PR CI.
    "tests/test_pipeline.py::TestSQLitePipelineSmartUpsert",
    # --- M18: the floorplan->OCR WIRING guard (daily-scrape.yml step ordering parsed as
    #     text — enrich-floorplans runs AFTER scrape + BEFORE ocr_enrich, into the DB OCR
    #     reads). Pins the fix for the OCR-starvation cliff. Pure file-parse (no DB/OCR/
    #     network/node — only TestWorkflowWiring, NOT the tesseract-gated OCR classes), so
    #     it always runs; if it went dark the enricher could be un-wired again unnoticed.
    "tests/test_floorplan_pipeline.py::TestWorkflowWiring",
    # --- OCR enrichment guards (pure decision logic — no-overwrite + sanity gate)
    "tests/test_ocr_enrich_guards.py",
    # --- Bad-extraction detector (pure regex/range logic)
    "tests/test_ocr_accuracy.py::TestBadExtractionDetector",
    # --- Mini rentals.db fixture wiring (the structural scrape-validation +
    #     backfill suites must keep EXECUTING in PR CI via the committed fixture, not
    #     silently all-skip when the live DB is absent). These are CI-safe: they read
    #     the committed fixture, no live DB/network/node.
    "tests/test_mini_db_fixture.py::test_committed_fixture_exists_and_nonempty",
    "tests/test_mini_db_fixture.py::test_fixture_has_pipeline_schema",
    "tests/test_mini_db_fixture.py::test_conftest_exposes_active_db_path",
    "tests/test_mini_db_fixture.py::test_scrape_validation_structural_tests_are_not_skipped",
]


def _nodeid_matches(collected: str, critical: str) -> bool:
    """A collected nodeid satisfies a critical entry if it equals it, is a
    parametrization of it (`…name[param]`), or lives under it (file / class prefix)."""
    if collected == critical:
        return True
    if collected.startswith(critical + "["):
        return True
    if collected.startswith(critical + "::"):
        return True
    # File-level entry (no `::`) — match any nodeid in that file.
    if "::" not in critical and collected.startswith(critical + "::"):
        return True
    return False


def _is_partial_run(config) -> bool:
    """True when pytest was invoked on a SUBSET (specific files/nodeids, or a -k filter).

    The allowlist enforcement only makes sense for a FULL-suite run (what CI does:
    `pytest -m "not live and not parity"`). A dev running `pytest tests/test_x.py` or
    `pytest -k foo` would otherwise see a confusing false failure because the critical
    tests simply weren't in that targeted session. Detect that and skip enforcement.

    Uses `config.args` — pytest's RESOLVED positional file/dir args (NOT the raw
    invocation flags, whose -m/-k VALUES would be misread as paths). On a bare run this
    is exactly the `testpaths` default (`['tests']`); a targeted run sets it to the
    specific path(s)/nodeid(s). A `-k` keyword expression also narrows the run.
    """
    args = list(getattr(config, "args", []) or [])
    # Default whole-suite invocation resolves to the testpaths root.
    default_roots = {"tests", "tests/", str(ROOT / "tests")}
    for a in args:
        if "::" in a:
            return True  # explicit nodeid
        if a.endswith(".py"):
            return True  # explicit file
        if a not in default_roots:
            return True  # some other narrower path
    if getattr(config.option, "keyword", None):
        return True  # -k expression narrows the run
    return False


def test_critical_tests_collected_and_not_skipped(request):
    """Every CRITICAL test must be COLLECTED and NOT skip-marked in this CI run.

    Enforced only on a FULL-suite run (CI). On a targeted dev run (`pytest tests/foo.py`
    or `-k expr`) the critical set isn't expected to be present, so we skip enforcement
    with a clear reason rather than false-alarm.
    """
    if _is_partial_run(request.config):
        pytest.skip(
            "partial/targeted pytest run — critical-test allowlist is enforced only on a "
            "full-suite run (CI runs `pytest -m \"not live and not parity\"`)."
        )

    session = request.session
    items = session.items  # all collected items (skips already marked by conftest)

    collected_ids = [it.nodeid for it in items]

    missing = []
    skipped = []
    for critical in CRITICAL_TESTS:
        matches = [it for it in items if _nodeid_matches(it.nodeid, critical)]
        if not matches:
            missing.append(critical)
            continue
        # Skipped == has a `skip`/`skipif` marker applied (conftest's no-DB/no-PG/no-OCR
        # skips, or a skipif on the test). If EVERY matching item is skip-marked, the
        # critical guard is providing zero protection in this env → fail loudly.
        def _is_skip_marked(it) -> bool:
            for m in it.iter_markers():
                if m.name == "skip":
                    return True
                if m.name == "skipif":
                    # skipif fires only if its condition is truthy.
                    if any(bool(c) for c in m.args):
                        return True
            return False

        if all(_is_skip_marked(it) for it in matches):
            skipped.append(critical)

    problems = []
    if missing:
        problems.append(
            "MISSING (not collected — renamed/moved/deleted/typo'd marker?):\n  - "
            + "\n  - ".join(missing)
        )
    if skipped:
        problems.append(
            "SKIPPED in CI (collected but skip-marked — an always-run guard went dark):\n  - "
            + "\n  - ".join(skipped)
        )

    assert not problems, (
        "ANTI-SILENT-SKIP GUARD TRIPPED — a critical regression test is not protecting "
        "PR CI.\n\n"
        + "\n\n".join(problems)
        + "\n\nIf you intentionally moved/renamed a critical test, update CRITICAL_TESTS "
        "in tests/test_ci_critical_tests_run.py in the SAME change. A critical test must "
        "stay CI-safe (no DB/OCR/Postgres/node deps) and run on every PR.\n"
        "Collected nodeids this run:\n  "
        + "\n  ".join(sorted(collected_ids))
    )


def test_allowlist_is_nonempty_and_well_formed():
    """Cheap self-check: the allowlist itself can't be silently emptied/garbled."""
    assert CRITICAL_TESTS, "CRITICAL_TESTS is empty — the silent-skip guard is disarmed."
    for entry in CRITICAL_TESTS:
        assert entry.startswith("tests/"), f"malformed critical nodeid: {entry!r}"
        assert "::" in entry or entry.endswith(".py"), f"malformed critical nodeid: {entry!r}"


# ---------------------------------------------------------------------------------
# A1 META-TEST — the daily-scrape DATA GATE must stay wired (CI-NOW).
#
# The single highest-leverage fix in Wave 1: daily-scrape.yml ran NO pytest at all, so
# a bad scrape / regressed model was committed + retrained with zero data-validation.
# A1 adds a `pytest -m data ...` step AFTER scrape/dedupe/OCR and BEFORE the commit/
# retrain step, gating the commit on it. This meta-test pins that activation so it
# cannot silently regress (the same failure class the whole module guards against, but
# for a WORKFLOW step rather than a collected test): if someone deletes the data-gate
# step or drops the `-m data` selector, THIS fails loudly in PR CI.
#
# It parses the YAML as TEXT (no PyYAML dep — same discipline as test_floorplan_pipeline
# TestWorkflowWiring) so it always runs in CI without an extra binary dependency.
# ---------------------------------------------------------------------------------
_DAILY_SCRAPE_YML = ROOT / ".github" / "workflows" / "daily-scrape.yml"


def test_daily_scrape_runs_pytest_data_gate():
    """daily-scrape.yml must invoke `pytest -m data` post-scrape (the activated A1 gate).

    Pins the single highest-leverage activation: the scheduled run validates the
    materialized DB with the `data`-marked suites before committing/retraining a model.
    Parses the workflow as text so it runs in PR CI with no PyYAML dependency.
    """
    assert _DAILY_SCRAPE_YML.exists(), (
        f"daily-scrape.yml missing at {_DAILY_SCRAPE_YML} — cannot verify the data gate."
    )
    text = _DAILY_SCRAPE_YML.read_text(encoding="utf-8")

    # Tolerant of quoting/spacing variants: `pytest -m data`, `pytest -m "data"`,
    # `python3 -m pytest -m data`, `pytest  -m   data`. The load-bearing assertion is
    # that a `-m`-selected pytest run keyed on the `data` marker exists in the workflow.
    # Require the actual INVOCATION form `python[3] -m pytest -m data`, NOT a bare
    # `pytest -m data` that also appears in the human-readable step `- name:` line —
    # otherwise removing the real command while keeping the step name would pass vacuously.
    pat = re.compile(r"python3?\s+-m\s+pytest\s+-m\s+[\"']?data[\"']?", re.IGNORECASE)
    assert pat.search(text), (
        "daily-scrape.yml does NOT invoke `pytest -m data` — the post-scrape DATA GATE "
        "(A1) is not wired. The scheduled run would commit/retrain a model with ZERO "
        "data validation. Add a `python3 -m pytest -m data ...` step AFTER scrape/dedupe/"
        "OCR and BEFORE the commit/retrain step, and gate the commit on it."
    )


def test_daily_scrape_data_gate_runs_before_commit_and_retrain():
    """The data gate must run BEFORE the model is committed AND before it is retrained.

    A gate that runs AFTER the commit/retrain protects nothing — the bad artifact is
    already on main. Step order == byte position in the (sequential) workflow file, so
    the `pytest -m data` invocation must precede both the `git commit ... model` step and
    the `retrain_canonical.py` invocation.
    """
    text = _DAILY_SCRAPE_YML.read_text(encoding="utf-8")

    # Require the actual INVOCATION form `python[3] -m pytest -m data`, NOT a bare
    # `pytest -m data` that also appears in the human-readable step `- name:` line —
    # otherwise removing the real command while keeping the step name would pass vacuously.
    pat = re.compile(r"python3?\s+-m\s+pytest\s+-m\s+[\"']?data[\"']?", re.IGNORECASE)
    m = pat.search(text)
    assert m, "no `pytest -m data` step found (see test_daily_scrape_runs_pytest_data_gate)"
    gate_pos = m.start()

    # Match the actual INVOCATION (`python3 retrain_canonical.py …`), not a passing
    # mention in a comment — the step ordering is what matters.
    retrain_m = re.search(r"python3?\s+retrain_canonical\.py", text)
    assert retrain_m, "expected the existing `python3 retrain_canonical.py` step in daily-scrape.yml"
    retrain_pos = retrain_m.start()
    assert gate_pos < retrain_pos, (
        "the `pytest -m data` gate must run BEFORE retrain_canonical.py so a bad scrape "
        f"is caught before the model is retrained on it (gate@{gate_pos} retrain@{retrain_pos})."
    )

    commit_pos = text.find("git commit")
    assert commit_pos >= 0, "expected the existing model-commit step in daily-scrape.yml"
    assert gate_pos < commit_pos, (
        "the `pytest -m data` gate must run BEFORE the model is committed/pushed "
        f"(gate@{gate_pos} commit@{commit_pos})."
    )


# =================================================================================
# WAVE 2 — CI-CONFIG META-TESTS (group CI-CONFIG, writer 4).
#
# These pin the Wave-2 CI hardening so it cannot silently regress, exactly like the A1
# data-gate meta-tests above. They parse ci.yml / the guard files as TEXT (no PyYAML /
# no node dep) so they always run in PR CI. Each one was proven RED before the matching
# ci.yml / guard / allowlist edit landed.
# =================================================================================
_CI_YML = ROOT / ".github" / "workflows" / "ci.yml"
_DASHBOARD_ROUTES_GUARD = ROOT / "dashboard" / "test" / "dashboard_routes_guard.mjs"

# The NEW Wave-2 dashboard route harnesses (Group SERVING, writer 1). These are the
# ACTUAL filenames SERVING created, mapped to the spec items:
#   A6  — route-HANDLER tests, split rental + sale (route_handler_test / _sale_test)
#   A7  — dual-SQL byte-equality (db.ts↔similarQuery.js, saleDb.ts↔saleSimilarQuery.js)
#   R6/A8 — serving query-bug DOCUMENTING (xfail) tests (no-space postcode + stale peer)
# CI-CONFIG owns wiring these into ci.yml's dashboard-routes job AND the guard's
# REQUIRED_HARNESSES so they can never silently skip. This list is kept in lockstep with
# the files SERVING lands (the guard's orphan check fails on any drift).
_WAVE2_ROUTE_HARNESSES = [
    "route_handler_test.mjs",        # A6 — rental route-handler invocation (real route.ts)
    "route_handler_sale_test.mjs",   # A6 — for-sale route-handler invocation (real route.ts)
    "dual_sql_equality_test.mjs",    # A7 — db.ts↔similarQuery.js byte-equality (+ sale)
    "serving_query_bug_doc_test.mjs",# R6/A8 — no-space-postcode + stale-peer xfails
]


def test_ci_python_cov_includes_for_sale():
    """M15 — ci.yml python-tests must measure coverage of the for_sale/ money-path.

    for_sale/ is a 2180-line vertical that was entirely unmeasured by --cov, so a
    coverage regression in the sale model/serving/CLI seam was invisible. Pin that
    `--cov=for_sale` is in the python-tests coverage scope.
    """
    text = _CI_YML.read_text(encoding="utf-8")
    assert re.search(r"--cov=for_sale\b", text), (
        "ci.yml python-tests does NOT include `--cov=for_sale` in its coverage scope — "
        "the for_sale/ vertical (2180 lines, the sale money-path) is unmeasured, so a "
        "coverage regression there is invisible. Add --cov=for_sale to the pytest "
        "--cov flags in the python-tests job."
    )


def test_min_executed_floor_raised_with_headroom():
    """M19 — the MIN_EXECUTED silent-collapse floor must be raised toward the real count.

    The original 120 was set when CI executed ~197; the suite now executes ~390+ in CI
    (no live DB / no tesseract). A floor of 120 would no longer catch a collapse that
    halves the suite. Require MIN_EXECUTED >= 360 (comfortable headroom below the real
    CI-executed count, well above the original 120 noise floor) so a silent collapse is
    actually caught. The value is read from ci.yml's inline python guard.
    """
    text = _CI_YML.read_text(encoding="utf-8")
    m = re.search(r"MIN_EXECUTED\s*=\s*(\d+)", text)
    assert m, "MIN_EXECUTED assignment not found in ci.yml inline executed-count guard."
    value = int(m.group(1))
    assert value >= 360, (
        f"MIN_EXECUTED={value} is too low to catch a silent suite collapse. CI now "
        f"executes ~390+ tests; raise the floor to >=360 (with headroom below the real "
        f"count) so a collapse to a handful of tests fails loudly."
    )


def test_dedupe_cross_source_remove_in_critical_tests():
    """M18 — the new Wave-2 dedupe cascade/backup safety test must be CI-critical.

    test_dedupe_cross_source_remove pins that the destructive cross-source remove path
    cascade-deletes price_history and routes through the _safe_delete guard. It is a
    load-bearing data-safety guard with no DB/network/node dep, so it must always run in
    PR CI — pin it on the allowlist so it can't silently go dark.
    """
    assert "tests/test_dedupe_cross_source_remove.py" in CRITICAL_TESTS, (
        "tests/test_dedupe_cross_source_remove.py (the dedupe cascade-delete + "
        "price_history-backup safety guard) is not on CRITICAL_TESTS — add it so the "
        "anti-silent-skip guard ensures it always runs in PR CI."
    )


def test_wave1_critical_guards_present():
    """M18 — confirm the Wave-1 additions stayed on the critical allowlist.

    test_fingerprint / the SQLite smart-upsert / the floorplan->OCR wiring guard were
    added in Wave 1; this pins that they did not silently fall off the allowlist.
    """
    expected = [
        "tests/test_fingerprint.py",
        "tests/test_pipeline.py::TestSQLitePipelineSmartUpsert",
        "tests/test_floorplan_pipeline.py::TestWorkflowWiring",
    ]
    missing = [e for e in expected if e not in CRITICAL_TESTS]
    assert not missing, (
        "Wave-1 critical guards dropped off CRITICAL_TESTS: " + ", ".join(missing)
    )


def test_wave2_route_harnesses_wired_into_ci_and_guard():
    """The NEW dashboard route harnesses (A6/A7/R6-A8) must be wired into BOTH ci.yml's
    dashboard-routes job AND the dashboard_routes_guard REQUIRED_HARNESSES.

    Any `*_test.mjs` on disk that the guard's REQUIRED_HARNESSES / ci.yml don't reference
    fails the guard as an ORPHAN. Pin the wiring here (text-parse, no node) so the
    serving harnesses can never silently skip while the job stays green.
    """
    ci_text = _CI_YML.read_text(encoding="utf-8")
    guard_text = _DASHBOARD_ROUTES_GUARD.read_text(encoding="utf-8")
    problems = []
    for h in _WAVE2_ROUTE_HARNESSES:
        if h not in ci_text:
            problems.append(f"{h} not invoked in ci.yml dashboard-routes job")
        if h not in guard_text:
            problems.append(f"{h} not in dashboard_routes_guard REQUIRED_HARNESSES")
    assert not problems, (
        "Wave-2 dashboard route harness wiring incomplete:\n  - "
        + "\n  - ".join(problems)
        + "\n(Group SERVING creates these dashboard/test/*_test.mjs files; CI-CONFIG "
        "wires them into ci.yml + the guard so they cannot silently skip.)"
    )


# =================================================================================
# INC2 PROD-SCRAPE — FOR-SALE GHA WORKFLOW CI WIRING (W2 + W3).
#
# The for-sale scheduled scrape writes PRODUCTION Neon `sale_listings` via the new
# scripts/sync_sales_to_postgres.py. Its safety is pinned by THREE tests:
#   * tests/test_for_sale_scrape_workflow.py — workflow meta-test (text-parse; on the
#     CRITICAL_TESTS allowlist, always runs in PR CI).
#   * tests/test_sale_data_layer_safety.py   — in-process (SqlitePgCursor) sync safety
#     (on the allowlist; always runs in PR CI).
#   * tests/test_sale_sync_real_pg.py        — the real-Postgres deepening. This one is
#     NOT on the allowlist because it skips without a PG service container (the allowlist's
#     not-skipped assertion would fail). Instead it is pinned by the DEDICATED `sale-sync-pg`
#     ci.yml job below, which provides the `postgres:16` service container that runs it for
#     real. This is the same pattern as the Wave-2 dashboard-routes / similar-query harness.
#
# These meta-asserts (W3) pin BOTH allowlist entries (W2) AND the real-PG job's existence,
# so the prod-write safety net cannot be silently deleted. Pure text-parse of ci.yml
# (no PyYAML / DB / PG / node) so they always run in PR CI.
# =================================================================================


def _ci_job_block(ci_text: str, job_name: str) -> str:
    """Return the TEXT of a single ci.yml job block (`<job_name>:` header → next top-level
    job header / EOF), so an assertion about a job's contents can't be vacuously satisfied
    by an identical string living in a DIFFERENT job (e.g. `image: postgres:16` also appears
    in the dashboard-routes job). Jobs are 2-space indented under `jobs:` and their steps are
    more deeply indented, so the next `^  <name>:` line at exactly that indent ends the block.
    """
    lines = ci_text.splitlines(keepends=True)
    header_re = re.compile(rf"^  {re.escape(job_name)}:\s*$")
    job_header_re = re.compile(r"^  [A-Za-z0-9_-]+:\s*$")
    start = None
    for i, line in enumerate(lines):
        if header_re.match(line):
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if job_header_re.match(lines[j]):
            end = j
            break
    return "".join(lines[start:end])


def test_for_sale_sync_critical_tests_in_allowlist():
    """W2 — the two CI-safe for-sale prod-scrape guards must be on CRITICAL_TESTS.

    The workflow meta-test (text-parse) and the in-process sync-safety test (SqlitePgCursor
    stand-in prod) have no DB/PG/node dep, so they must always run in PR CI — pin them on the
    allowlist so the anti-silent-skip guard ensures the prod-write safety net stays armed.
    """
    expected = [
        "tests/test_for_sale_scrape_workflow.py",
        "tests/test_sale_data_layer_safety.py",
    ]
    missing = [e for e in expected if e not in CRITICAL_TESTS]
    assert not missing, (
        "for-sale prod-scrape critical guards missing from CRITICAL_TESTS: "
        + ", ".join(missing)
        + ".\nAdd them so the workflow-safety meta-test and the in-process sync-safety test "
        "always run in PR CI (they are CI-safe: text-parse / in-memory sqlite, no PG/node)."
    )


def test_sale_sync_real_pg_not_in_allowlist():
    """W2 — the real-Postgres sale-sync test must NOT be on the allowlist.

    tests/test_sale_sync_real_pg.py skips when POSTGRES_TEST_URL is unset (devs without a
    PG container aren't blocked). Putting it on CRITICAL_TESTS would make the allowlist's
    not-skipped assertion FAIL on the standard PR lane (no PG container). It is pinned
    instead by the dedicated `sale-sync-pg` ci.yml job (see the W3 meta-assert below), which
    supplies a postgres:16 service container so it runs for real.
    """
    assert "tests/test_sale_sync_real_pg.py" not in CRITICAL_TESTS, (
        "tests/test_sale_sync_real_pg.py is on CRITICAL_TESTS but it REQUIRES a Postgres "
        "service container (it pytest.skips without POSTGRES_TEST_URL). On the standard PR "
        "lane it would be collected-but-skipped, tripping the anti-silent-skip allowlist "
        "assertion. Remove it from CRITICAL_TESTS — it is pinned by the dedicated "
        "`sale-sync-pg` ci.yml job instead (test_ci_has_sale_sync_pg_job)."
    )


def test_ci_has_sale_sync_pg_job():
    """W3 — ci.yml must keep the dedicated `sale-sync-pg` real-Postgres job.

    The real-PG sale-sync safety test (tests/test_sale_sync_real_pg.py) is deliberately OFF
    the CRITICAL_TESTS allowlist (it skips without a DB), so the ONLY thing keeping it from
    silently disappearing is this job. Pin, WITHIN the `sale-sync-pg` job block (so an
    identical string in another job — `image: postgres:16` also lives in dashboard-routes —
    can't satisfy it vacuously), that the job:
      * exists as a top-level ci.yml job named `sale-sync-pg`,
      * runs a `postgres:16` service container (so the test executes for REAL, not skipped),
      * invokes `pytest tests/test_sale_sync_real_pg.py`.
    Mirrors the INVOCATION + isolation discipline of test_ci_python_cov_includes_for_sale /
    test_wave2_route_harnesses_wired_into_ci_and_guard. Pure text-parse (no PyYAML/PG/node).
    """
    ci_text = _CI_YML.read_text(encoding="utf-8")

    assert re.search(r"^  sale-sync-pg:\s*$", ci_text, re.MULTILINE), (
        "ci.yml has NO top-level `sale-sync-pg` job. The real-Postgres sale-sync safety "
        "test (tests/test_sale_sync_real_pg.py) is OFF the CRITICAL_TESTS allowlist (it "
        "skips without a PG container), so this dedicated job is the ONLY thing that runs "
        "it. Add the `sale-sync-pg` job (mirror the dashboard-routes service-container block)."
    )

    block = _ci_job_block(ci_text, "sale-sync-pg")
    assert block, "could not isolate the `sale-sync-pg` job block in ci.yml."

    problems = []
    if not re.search(r"image:\s*postgres:16\b", block):
        problems.append(
            "the `sale-sync-pg` job does not declare an `image: postgres:16` service "
            "container — the real-PG test would skip (no POSTGRES_TEST_URL) instead of "
            "exercising the prod-write UPSERT/backup/delta-abort path for real."
        )
    if not re.search(r"pytest\s+tests/test_sale_sync_real_pg\.py", block):
        problems.append(
            "the `sale-sync-pg` job does not invoke "
            "`pytest tests/test_sale_sync_real_pg.py` — the real-Postgres sale-sync safety "
            "assertions (CREATE-if-absent, non-destructive UPSERT, first_seen-never, "
            "backup-first, delta-abort, conflict-key refusal) would never run."
        )
    assert not problems, (
        "`sale-sync-pg` ci.yml job is present but incomplete:\n  - "
        + "\n  - ".join(problems)
        + "\n(This is the ONLY net for tests/test_sale_sync_real_pg.py, which is off the "
        "CRITICAL_TESTS allowlist by design — it must run against a real postgres:16 "
        "service container.)"
    )
