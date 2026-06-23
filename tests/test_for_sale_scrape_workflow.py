"""Test B — WORKFLOW META-TEST for the PRODUCTION for-sale scrape workflow.

WHY THIS EXISTS
---------------
.github/workflows/for-sale-scrape.yml writes PRODUCTION Neon `sale_listings`. Three of
its safety properties are STRUCTURALLY uncatchable by anything except a text-parse of the
workflow YAML itself, and would vanish *silently* if the file were edited:

  * SCHEDULED + DISPATCH ONLY — if someone added a `push:`/`pull_request:` trigger, merging
    the PR would AUTO-FIRE a prod write. Nothing else in CI would notice.
  * SYNC IS GATED — the prod-write sync must pass BOTH `--execute` AND
    `--i-have-rotated-the-secret`; drop one token and the gate is gone.
  * ISOLATION — the sale workflow must touch ONLY sale_listings / sales.db /
    sync_sales_to_postgres.py; it must never reach into rentals.db, the rental `listings`
    table, retrain_canonical, the rental sync, or a model git-push.

These meta-tests run on EVERY PR (pure text-parse — no PyYAML, no DB, no node), exactly
like tests/test_live_smoke_workflows.py, and FAIL LOUDLY the moment any of those properties
regress. This file is on the CRITICAL_TESTS allowlist.

Mirrors tests/test_live_smoke_workflows.py: the _read (existence-asserting) and _on_block
(isolated top-level `on:` block) helpers, so a trigger elsewhere in the file cannot
accidentally satisfy/violate the scheduled-only check.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / ".github" / "workflows"
FOR_SALE_YML = WF / "for-sale-scrape.yml"
DAILY_YML = WF / "daily-scrape.yml"

# The single load-bearing sync invocation: the REAL `python3 scripts/sync_sales_to_postgres.py`
# command line, NOT a `- name:` step label that merely mentions it (mirror the INVOCATION
# discipline in test_ci_critical_tests_run.py — a step name proves nothing runs).
SYNC_INVOCATION = "python3 scripts/sync_sales_to_postgres.py"
SCRAPE_INVOCATION = "python -m cli.main scrape"


def _read(p: Path) -> str:
    assert p.exists(), (
        f"{p.relative_to(ROOT)} is MISSING — the for-sale prod-scrape workflow is gone."
    )
    return p.read_text(encoding="utf-8")


def _on_block(text: str) -> str:
    """The top-level `on:` block only (up to the next top-level key), so a `pull_request`
    elsewhere in the file can't accidentally satisfy/violate the trigger checks."""
    m = re.search(r"(?ms)^on:\s*\n(.*?)^[A-Za-z]", text)
    assert m, "workflow has no top-level `on:` block."
    return m.group(1)


def _invocation_lines(text: str, needle: str) -> list[str]:
    """Lines that actually RUN the command (not a `- name:`/comment mention of it)."""
    out = []
    for line in text.splitlines():
        if needle in line and not line.lstrip().startswith("#") and "- name:" not in line:
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# B1 — EXISTS
# ---------------------------------------------------------------------------
def test_b1_workflow_exists():
    text = _read(FOR_SALE_YML)
    assert text.strip(), "for-sale-scrape.yml is empty."


# ---------------------------------------------------------------------------
# B2 — SCHEDULED + DISPATCH ONLY (constraint #3 — merging must not auto-scrape prod)
# ---------------------------------------------------------------------------
def test_b2_scheduled_and_dispatch_only_no_push_pr():
    text = _read(FOR_SALE_YML)
    on = _on_block(text)

    # Scheduled cron — a prod scrape that never fires is useless.
    assert "schedule:" in on and "cron:" in on, (
        "for-sale-scrape.yml must have a `schedule:`/`cron:` trigger."
    )
    # Manual dispatch — the lead's first prod run is a deliberate workflow_dispatch.
    assert "workflow_dispatch" in on, (
        "for-sale-scrape.yml must support `workflow_dispatch` for the gated first prod run."
    )
    # NO push/pull_request anywhere in the on-block — merging the PR that adds this file
    # must NOT auto-fire a prod write (constraint #3). Use _on_block so a `push`/`pull_request`
    # token elsewhere (comments, job names) can't satisfy or violate this.
    assert not re.search(r"(?m)^\s*(push|pull_request)\s*:", on), (
        "for-sale-scrape.yml must NOT have a push/pull_request trigger — merging it would "
        "auto-scrape + write PROD Neon. Scheduled + workflow_dispatch ONLY."
    )

    # The dispatch inputs the workflow contract promises (text-parse, not schema-parse).
    assert "max_pages" in text, "workflow_dispatch must expose a `max_pages` input."
    assert "execute_sync" in text, "workflow_dispatch must expose an `execute_sync` input."


# ---------------------------------------------------------------------------
# B3 — CONCURRENCY (distinct group so sale never collides with itself or rental)
# ---------------------------------------------------------------------------
def test_b3_concurrency_group_present_and_distinct():
    text = _read(FOR_SALE_YML)
    assert re.search(r"(?m)^concurrency:", text), (
        "for-sale-scrape.yml must declare a top-level `concurrency:` block."
    )
    assert "for-sale-scrape" in text, (
        "concurrency group must be the distinct `for-sale-scrape` (not rental's)."
    )
    # A mid-flight prod sync must never be cancelled.
    assert re.search(r"cancel-in-progress:\s*false", text), (
        "concurrency must set cancel-in-progress: false — never cancel a mid-flight prod sync."
    )


# ---------------------------------------------------------------------------
# B4 — SYNC INVOCATION + GATING (the load-bearing prod-write safety check)
# ---------------------------------------------------------------------------
def test_b4_sync_invocation_is_gated_and_uses_secret():
    text = _read(FOR_SALE_YML)

    # The REAL invocation line must exist (not just a `- name:` mention).
    invs = _invocation_lines(text, SYNC_INVOCATION)
    assert invs, (
        f"for-sale-scrape.yml must actually RUN `{SYNC_INVOCATION}` (a `- name:` mention "
        "is not enough — mirror the INVOCATION discipline)."
    )

    # At least one invocation must be the gated PROD write carrying BOTH confirmation tokens
    # on the same command line (a dry-run invocation may exist too, but the executing one
    # must be fully gated).
    gated = [
        ln for ln in invs
        if "--execute" in ln and "--i-have-rotated-the-secret" in ln
    ]
    assert gated, (
        "the executing sync invocation must pass BOTH `--execute` AND "
        "`--i-have-rotated-the-secret` on the same line — the prod-write double gate."
    )

    # The prod secret must be referenced (via env / inline). Mirrors daily-scrape.yml:22.
    assert "${{ secrets.POSTGRES_URL }}" in text, (
        "for-sale-scrape.yml must reference ${{ secrets.POSTGRES_URL }} (the prod Neon secret)."
    )


# ---------------------------------------------------------------------------
# B5 — NO DESTRUCTIVE SQL anywhere in the YAML body
# ---------------------------------------------------------------------------
def test_b5_no_destructive_sql_in_workflow_body():
    text = _read(FOR_SALE_YML)
    upper = text.upper()
    for bad in ("TRUNCATE", "DROP TABLE", "DELETE FROM SALE_LISTINGS", "--TRUNCATE", "SETVAL("):
        assert bad not in upper, (
            f"for-sale-scrape.yml must contain NO destructive SQL — found {bad!r}. "
            "All prod writes go through the UPSERT-only, shrink-aborting sync script."
        )


# ---------------------------------------------------------------------------
# B6 — ORDER: scrape (sale, no --postgres, no --full) PRECEDES the sync
# ---------------------------------------------------------------------------
def test_b6_scrape_sale_precedes_sync_and_has_no_postgres_no_full():
    text = _read(FOR_SALE_YML)

    scrape_invs = _invocation_lines(text, SCRAPE_INVOCATION)
    assert scrape_invs, f"for-sale-scrape.yml must run `{SCRAPE_INVOCATION}`."

    # The scrape must select sale mode.
    assert "--listing-type sale" in text, (
        "the scrape step must pass `--listing-type sale` (selects the for-sale path)."
    )

    # NO --postgres on the scrape line (cli/main.py hard-exits on sale+--postgres) and
    # NO --full (it triggers the rental-only enrich/dedupe tail against rentals.db).
    for ln in scrape_invs:
        assert "--postgres" not in ln, (
            "the sale scrape must NOT pass --postgres — cli.main hard-exits on sale+--postgres."
        )
        assert "--full" not in ln, (
            "the sale scrape must NOT pass --full — it fires the rental-only enrich/dedupe "
            "tail against rentals.db (cross-vertical contamination)."
        )

    # Byte-position ordering: the scrape invocation precedes the sync invocation.
    scrape_pos = text.index(scrape_invs[0])
    sync_pos = text.index(_invocation_lines(text, SYNC_INVOCATION)[0])
    assert scrape_pos < sync_pos, (
        "the scrape step must PRECEDE the sync step — you can't sync sales.db before it's "
        "been scraped."
    )


# ---------------------------------------------------------------------------
# B7 — PLAYWRIGHT INSTALL (savills/knightfrank/chestertons sale spiders are Playwright)
# ---------------------------------------------------------------------------
def test_b7_playwright_chromium_installed():
    text = _read(FOR_SALE_YML)
    assert "playwright install chromium" in text, (
        "for-sale-scrape.yml must install Playwright chromium — the sale spiders "
        "(savills/knightfrank/chestertons) are Playwright-driven."
    )


# ---------------------------------------------------------------------------
# B8 — ISOLATION (constraint #4 — zero cross-vertical reach)
# ---------------------------------------------------------------------------
def test_b8_isolation_no_rental_references():
    """RENTAL-isolation invariant: the sale workflow must never reach into the rental
    vertical. The workflow now retrains + commits the SALE model, so the sale-scoped
    tokens `fixture_diff` (node chrome-extension/sale_fixture_diff.mjs) and `git push`
    (the SALE model commit step) are LEGITIMATE and explicitly allowed — they operate on
    sale_* artifacts only. The rental tokens below stay HARD-FORBIDDEN."""
    text = _read(FOR_SALE_YML)
    forbidden = [
        "rentals.db",
        "FROM listings",
        "retrain_canonical",
        "--listing-type rent",
        "enrich-floorplans",
        "ocr_enrich",
        "sync_sqlite_to_postgres.py",
    ]
    for token in forbidden:
        assert token not in text, (
            f"for-sale-scrape.yml must NOT reference {token!r} — it touches ONLY the SALE "
            "vertical (sales.db / sale_listings / sync_sales_to_postgres.py + sale_* model "
            "artifacts); it must never reach the rental DB, table, retrain, or sync "
            "(constraint #4)."
        )

    # The sale workflow legitimately uses the SALE-scoped parity gate and a SALE model
    # commit; assert those are the *sale* variants, never the rental ones.
    if "fixture_diff" in text:
        assert "sale_fixture_diff.mjs" in text, (
            "the only `fixture_diff` reference allowed is the SALE parity gate "
            "`chrome-extension/sale_fixture_diff.mjs` — never the rental fixture_diff.mjs."
        )
        assert "chrome-extension/fixture_diff.mjs" not in text, (
            "the RENTAL fixture_diff.mjs must never appear in the sale workflow."
        )


# ---------------------------------------------------------------------------
# B8b — SALE RETRAIN + COMMIT CONTRACT (the new §3 W0 contract this workflow now carries)
# ---------------------------------------------------------------------------
def test_b8b_sale_retrain_and_commit_contract():
    """After W0 the workflow retrains the SALE model on the REAL sales.db and commits the
    served matched set back to main. These are the load-bearing properties of that contract;
    they would vanish silently on an edit, so the meta-test locks them."""
    text = _read(FOR_SALE_YML)

    # contents: write (mirrors daily-scrape) — required to push the model commit.
    assert re.search(r"(?m)^\s*contents:\s*write\b", text), (
        "for-sale-scrape.yml must declare `contents: write` — it now commits the SALE "
        "model matched set back to main."
    )

    # The REAL-data retrain must actually run (invocation, not a `- name:` mention) and
    # train the sale model.
    assert ("run_sale_retrain" in text) or ("for_sale.sale_retrain" in text), (
        "for-sale-scrape.yml must RUN the sale retrain (for_sale.sale_retrain / "
        "run_sale_retrain) on the real sales.db."
    )

    # Synthetic-leak detector: the workflow gate asserts the model was trained on real data.
    assert 'db_source"].endswith("sales.db")' in text or 'endswith("sales.db")' in text, (
        "the retrain step must assert meta.db_source endswith 'sales.db' — the "
        "synthetic-leak detector (a synthetic run ends in 'sale_training_sample.json')."
    )
    # Quality band on R2 so a degenerate model never ships.
    assert "MIN_REAL_SALE_ROWS" in text, (
        "the retrain gate must assert n_train >= MIN_REAL_SALE_ROWS."
    )
    assert "0.999" in text and "0.5" in text, (
        "the retrain gate must assert the R2 sane band 0.5 <= r2 <= 0.999."
    )

    # The parity gate must run BOTH legs (JS==inference + the sale fixture diff).
    assert "sync_sale_js_freq_maps.py --check" in text, (
        "the parity gate must run `sync_sale_js_freq_maps.py --check` (JS == inference)."
    )
    assert "node chrome-extension/sale_fixture_diff.mjs" in text, (
        "the parity gate must run `node chrome-extension/sale_fixture_diff.mjs` (0/0 gate)."
    )

    # Node toolchain for the parity gate.
    assert "actions/setup-node" in text, (
        "for-sale-scrape.yml must set up Node (actions/setup-node) — the parity gate runs "
        "`node chrome-extension/sale_fixture_diff.mjs`."
    )

    # COMMIT step: explicit named `git add` of the SALE matched set only — never `git add -A`,
    # never the runner's populated sales.db.
    assert "git push" in text, (
        "the workflow must `git push` the committed SALE model matched set."
    )
    assert "git add -A" not in text and "git add ." not in text, (
        "the commit step must use explicit named `git add` paths — NEVER `git add -A`/"
        "`git add .` (the runner's sales.db / logs must never be committed)."
    )
    for art in (
        "output/sale_api/model.json",
        "output/sale_api/features.json",
        "output/sale_model_inference.json",
        "output/sale_feature_parity_golden.json",
        "chrome-extension/sale_xgboost.js",
        "dashboard/src/app/api/predict-sale/sale_xgboost.predictor.js",
        "output/sale_model.pkl",
        "output/sale_model_meta.json",
    ):
        assert art in text, (
            f"the commit step must `git add` the SALE artifact {art!r} (the served matched set)."
        )
    # The gitignored pkls must be force-added.
    assert "git add -f" in text and "output/sale_model.pkl" in text, (
        "the gitignored sale pkls must be `git add -f`'d."
    )
    # NEVER commit the runner's populated DB.
    assert not re.search(r"git add[^\n]*output/sales\.db", text), (
        "the commit step must NEVER `git add output/sales.db` — the runner's populated DB "
        "must stay out of the commit."
    )


# ---------------------------------------------------------------------------
# B9 — DAILY-SCRAPE UNCHANGED (belt-and-braces: nobody bolted sale onto rental)
# ---------------------------------------------------------------------------
def test_b9_daily_scrape_has_no_sale_steps():
    text = _read(DAILY_YML)
    for token in ("sale_listings", "--listing-type sale", "sync_sales_to_postgres"):
        assert token not in text, (
            f"daily-scrape.yml must contain NO {token!r} — the rental workflow must stay "
            "byte-unchanged; sale lives in its own workflow (constraint #4)."
        )


# ---------------------------------------------------------------------------
# B10 — NOTIFY-FAILURE NON-BLOCKING (distinct label from rental)
# ---------------------------------------------------------------------------
def test_b10_notify_failure_job_present_and_distinct_label():
    text = _read(FOR_SALE_YML)
    assert "if: failure()" in text, (
        "for-sale-scrape.yml must have an `if: failure()` notify job."
    )
    assert "actions/github-script" in text or "issues.create" in text, (
        "the failure job must open/refresh a GitHub issue (github-script / issues.create)."
    )
    assert "for-sale-scrape-failure" in text, (
        "the failure issue must be labelled `for-sale-scrape-failure` (distinct from rental's "
        "`scrape-failure`)."
    )
