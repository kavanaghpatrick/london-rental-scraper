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
    # --- Data-layer safety (the destructive-op guards / _safe_delete)
    "tests/test_safe_delete.py",
    "tests/test_data_layer_safety.py",
    # --- Cycle-relative mark-inactive (the recurring nightly prod is_active wipe).
    #     Pure file-parse + in-memory SQLite — no DB/PG/node. Pins the SHIPPED
    #     daily-scrape.yml SQL to MAX(last_seen), not wall-clock NOW(); must never
    #     silently stop running, or the wall-clock-wipe footgun could return unguarded.
    "tests/test_mark_inactive_cycle_relative.py",
    # --- Cross-source dedupe identity core (pure, no DB)
    "tests/test_dedupe_postgres.py::test_distinct_streets_not_merged",
    "tests/test_dedupe_postgres.py::test_true_cross_source_dupe_still_caught",
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
