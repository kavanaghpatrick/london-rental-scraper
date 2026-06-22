"""Meta-tests pinning the Wave-3 LIVE-SMOKE layer so it cannot silently disappear.

WHY THIS EXISTS
---------------
Some failure classes are STRUCTURALLY uncatchable by headless PR CI: a live site silently
migrating its DOM/JSON embedding format (the class that shipped the for-sale extension
broken), and prod Postgres schema drifting away from the SQL casts the routes assume.
The fix is two SCHEDULED, NON-PR-BLOCKING workflows that open a GitHub issue on failure
rather than gating merges:
  * .github/workflows/extraction-drift-smoke.yml  — weekly live per-site extraction smoke
  * .github/workflows/prod-schema-drift.yml       — weekly read-only prod information_schema check

Because those jobs never run in the PR path, nothing in PR CI would notice if one were
deleted, lost its schedule trigger, gained a merge-gating trigger, or dropped its
issue-on-failure step — the safety net would vanish silently. THESE meta-tests DO run on
every PR (they parse the YAML/py as text — no PyYAML dependency, no DB/node) and FAIL
LOUDLY if that happens. They are on the CRITICAL_TESTS allowlist.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / ".github" / "workflows"
EXTRACTION_YML = WF / "extraction-drift-smoke.yml"
SCHEMA_YML = WF / "prod-schema-drift.yml"
SCHEMA_SCRIPT = ROOT / "scripts" / "check_prod_schema_drift.py"


def _read(p: Path) -> str:
    assert p.exists(), f"{p.relative_to(ROOT)} is MISSING — the live-smoke layer it belongs to is gone."
    return p.read_text(encoding="utf-8")


def _on_block(text: str) -> str:
    """The top-level `on:` block only (up to the next top-level key), so a `pull_request`
    elsewhere in the file can't accidentally satisfy/violate the trigger checks."""
    m = re.search(r"(?ms)^on:\s*\n(.*?)^[A-Za-z]", text)
    assert m, "workflow has no top-level `on:` block."
    return m.group(1)


@pytest.mark.parametrize(
    "path,label,refers",
    [
        (EXTRACTION_YML, "extraction-drift", "headless_extraction_smoke.mjs"),
        (SCHEMA_YML, "schema-drift", "check_prod_schema_drift.py"),
    ],
)
def test_live_smoke_workflow_scheduled_nonblocking_issue_on_failure(path, label, refers):
    text = _read(path)
    on = _on_block(text)

    # 1. SCHEDULED — must run on a cadence (a live-smoke that never fires protects nothing).
    assert "schedule:" in on and "cron:" in on, (
        f"{path.name} must have a `schedule:`/`cron:` trigger — a live-smoke must run weekly."
    )

    # 2. NON-PR-BLOCKING — must NOT have a push/pull_request trigger that gates merges.
    assert not re.search(r"(?m)^\s*(push|pull_request)\s*:", on), (
        f"{path.name} must NOT have a push/pull_request trigger — live-smokes are network-flaky "
        "and must signal (open an issue), not red-gate merges."
    )

    # 3. ISSUE-ON-FAILURE — a job that opens/updates a GitHub issue when the smoke fails.
    assert "if: failure()" in text, (
        f"{path.name} must have an `if: failure()` job that surfaces drift."
    )
    assert "actions/github-script" in text or "issues.create" in text, (
        f"{path.name} must create/update a GitHub issue on failure (github-script / issues.create)."
    )
    assert label in text, f"{path.name} must label its drift issue '{label}'."

    # 4. Must actually invoke its harness/script (not an empty scaffold).
    assert refers in text, f"{path.name} must invoke {refers}."


def test_prod_schema_drift_is_read_only_and_skips_without_secret():
    src = _read(SCHEMA_SCRIPT)

    # The real guarantee: the connection is opened READ-ONLY at the DB level, so no SQL
    # (whatever it is) can mutate prod.
    assert re.search(r"set_session\(\s*readonly\s*=\s*True", src), (
        "check_prod_schema_drift.py must open a READ-ONLY connection (set_session(readonly=True)) "
        "— it touches prod Neon and must never write."
    )

    # Belt-and-braces: no write/DDL SQL in the executable body (strip the module docstring
    # + `#` comments first, since those legitimately mention CREATE TABLE etc. by name).
    no_docstrings = re.sub(r'(?s)""".*?"""', "", src)
    no_docstrings = re.sub(r"(?s)'''.*?'''", "", no_docstrings)
    code = "\n".join(line.split("#", 1)[0] for line in no_docstrings.splitlines())
    for bad in ("INSERT INTO", "DELETE FROM", "TRUNCATE", "DROP TABLE", "ALTER TABLE", "CREATE TABLE", "UPDATE "):
        assert bad not in code.upper(), (
            f"check_prod_schema_drift.py executable body must contain NO write SQL — found {bad!r}."
        )

    # Graceful skip on a fork / unconfigured env (no Neon secret) — never a spurious failure.
    assert "skipped" in src and re.search(r"(?i)no[_ ].*(url|secret)|getenv|environ", src), (
        "check_prod_schema_drift.py must SKIP gracefully (exit 0) when the Neon secret is absent."
    )
