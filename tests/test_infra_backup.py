"""
Infra/backup tests (#7 / Problem 2): the Neon prod backup failed DAILY because CI
installed postgresql-client **16** while the Neon server is **17.10**, so pg_dump
aborted with "server version mismatch". backup_neon.sh's `--check` preflight
checked only that pg_dump EXISTS, never that its major matched the server.

This file pins the WHOLE fix in two layers:

A. The major-version preflight engine + gate (genuine non-mock Red/Green):
  1. parse_major() — pure unit, no binary, always runs in CI. (the engine)
  2. scripts/pg_version.py --check-major — exercised against REAL pg16/pg17
     binaries when present; skips a major it can't find.
  3. backup_neon.sh --check — the integration gate: with NEON_SERVER_MAJOR=17 it
     must FAIL loudly when pointed at a pg16 pg_dump and PASS for a pg17 pg_dump.

B. Static guards the PLAN (docs/PIPELINE_FIX_PLAN.md, P3) named explicitly:
  - test_neon_backup_installs_pg17  — the workflow installs postgresql-client-17
    via PGDG and NOT the bare (=16) client.
  - test_backup_script_preflight_check — `backup_neon.sh --check` exit-0 contract
    (skipif no pg_dump on PATH).
  - test_backup_script_hint_text_says_17 — the script's hint text no longer says
    "PostgreSQL 16".

The binary-dependent cases (A2/A3) `skip` only their own case when a major is
absent, so the suite is green wherever it runs while still proving Red/Green on a
box that has both (e.g. local Homebrew pg16+pg17).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import pg_version  # noqa: E402

BACKUP_SH = REPO / "scripts" / "backup_neon.sh"
PG_VERSION_PY = REPO / "scripts" / "pg_version.py"

# Candidate locations for real binaries of each major. PATH `pg_dump` is tried
# first (that's what CI installs); Homebrew kegs cover local dev (pg16 + pg17).
_HOMEBREW = {
    16: "/opt/homebrew/opt/postgresql@16/bin/pg_dump",
    17: "/opt/homebrew/opt/postgresql@17/bin/pg_dump",
    18: "/opt/homebrew/opt/postgresql@18/bin/pg_dump",
}


def _find_pg_dump(major: int) -> str | None:
    """Return a path to a pg_dump binary whose major == `major`, or None."""
    candidates = []
    path_pg = shutil.which("pg_dump")
    if path_pg:
        candidates.append(path_pg)
    hb = _HOMEBREW.get(major)
    if hb:
        candidates.append(hb)
    for c in candidates:
        if not c or not os.path.exists(c):
            continue
        try:
            if pg_version.pg_dump_major(c) == major:
                return c
        except pg_version.PgVersionError:
            continue
    return None


# ---------------------------------------------------------------------------
# 1. Pure unit tests for the parser — no binary, always run (CI-gated).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("pg_dump (PostgreSQL) 17.9 (Homebrew)", 17),
        ("pg_dump (PostgreSQL) 17.10", 17),
        ("pg_dump (PostgreSQL) 16.14 (Ubuntu 16.14-1.pgdg24.04+1)", 16),
        ("pg_dump (PostgreSQL) 16.14 (Homebrew)", 16),
        ("pg_dump (PostgreSQL) 18.0 (Debian 18.0-1.pgdg13+3)", 18),
        ("pg_dump (PostgreSQL) 10.23", 10),
    ],
)
def test_parse_major_handles_all_vendor_forms(text, expected):
    assert pg_version.parse_major(text) == expected


def test_parse_major_rejects_garbage():
    with pytest.raises(pg_version.PgVersionError):
        pg_version.parse_major("not a version string")


def test_parse_major_rejects_empty():
    with pytest.raises(pg_version.PgVersionError):
        pg_version.parse_major("")


# ---------------------------------------------------------------------------
# 2. The CLI gate (scripts/pg_version.py --check-major) against REAL binaries.
#    This is the genuine non-mock Red/Green: pg16 must FAIL the 17-gate, pg17 PASS.
# ---------------------------------------------------------------------------

def _run_check_major(pg_dump: str, expected_major: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PG_VERSION_PY), "--check-major", str(expected_major),
         "--pg-dump", pg_dump],
        capture_output=True, text=True, timeout=30,
    )


def test_check_major_FAILS_for_pg16_against_server17():
    """RED reproduction: a v16 client must be REJECTED for a v17 server."""
    pg16 = _find_pg_dump(16)
    if not pg16:
        pytest.skip("no postgresql-client-16 pg_dump available on this host")
    cp = _run_check_major(pg16, 17)
    assert cp.returncode == 1, (
        f"pg16 should FAIL the v17 gate but returncode={cp.returncode}\n{cp.stderr}"
    )
    assert "mismatch" in cp.stderr.lower() or "!=" in cp.stderr


def test_check_major_PASSES_for_pg17_against_server17():
    """GREEN: a v17 client matches a v17 server."""
    pg17 = _find_pg_dump(17)
    if not pg17:
        pytest.skip("no postgresql-client-17 pg_dump available on this host")
    cp = _run_check_major(pg17, 17)
    assert cp.returncode == 0, (
        f"pg17 should PASS the v17 gate but returncode={cp.returncode}\n{cp.stderr}"
    )


# ---------------------------------------------------------------------------
# 3. The integration gate: backup_neon.sh --check must enforce the major.
#    Uses a dummy --url so --check never touches a real server.
# ---------------------------------------------------------------------------

_DUMMY_URL = "postgresql://u:p@dummy-host.example/db"


def _run_backup_check(pg_dump: str, server_major: int) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NEON_SERVER_MAJOR"] = str(server_major)
    # Point the script at a chosen pg_dump by prepending its dir to PATH.
    env["PATH"] = str(Path(pg_dump).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        ["bash", str(BACKUP_SH), "--check", "--url", _DUMMY_URL],
        capture_output=True, text=True, timeout=30, env=env,
    )


def test_backup_check_FAILS_when_pg16_vs_server17():
    """The --check preflight must catch client-16-vs-server-17 (the daily failure)."""
    pg16 = _find_pg_dump(16)
    if not pg16:
        pytest.skip("no postgresql-client-16 pg_dump available on this host")
    cp = _run_backup_check(pg16, 17)
    assert cp.returncode != 0, (
        "backup_neon.sh --check must FAIL when pg_dump major (16) != server major (17), "
        f"but it exited 0.\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
    )
    combined = (cp.stdout + cp.stderr).lower()
    assert "major" in combined or "mismatch" in combined or "version" in combined


def test_backup_check_PASSES_when_pg17_vs_server17():
    """--check must succeed when the client major matches the server major."""
    pg17 = _find_pg_dump(17)
    if not pg17:
        pytest.skip("no postgresql-client-17 pg_dump available on this host")
    cp = _run_backup_check(pg17, 17)
    assert cp.returncode == 0, (
        "backup_neon.sh --check should PASS when pg_dump major (17) == server major (17), "
        f"but it exited {cp.returncode}.\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
    )


# ---------------------------------------------------------------------------
# B. Static guards named by docs/PIPELINE_FIX_PLAN.md (P3). These bite the
#    "16" install / stale hints today (RED) and lock the fix (GREEN).
# ---------------------------------------------------------------------------

NEON_BACKUP_YML = REPO / ".github" / "workflows" / "neon-backup.yml"


def test_neon_backup_installs_pg17():
    """neon-backup.yml must install postgresql-client-17 via PGDG, not bare client-16.

    RED today: the step installed `apt-get install -y postgresql-client` (the
    distro default = 16). GREEN: pins `postgresql-client-17` from the PGDG repo.
    """
    yml = NEON_BACKUP_YML.read_text()
    # Must reference the version-pinned client (package or versioned bindir).
    assert (
        "postgresql-client-17" in yml or "/usr/lib/postgresql/17/bin" in yml
    ), "neon-backup.yml must install postgresql-client-17 (matching Neon server 17)."
    # Must use the PGDG repo (Ubuntu's default repo tops out at 16).
    assert (
        "apt.postgresql.org" in yml or "pgdg" in yml.lower()
    ), "neon-backup.yml must add the PGDG apt repo to get pg client 17."
    # Must NOT install the bare (=16) client. Allow only the *-17 / *-common forms.
    bare = re.findall(r"install[^\n]*\bpostgresql-client\b(?!-)", yml)
    assert not bare, (
        f"neon-backup.yml still installs the bare postgresql-client (=16 on Ubuntu): {bare}"
    )


def test_backup_script_preflight_check():
    """`backup_neon.sh --check` honours its exit-0 contract with a matching client.

    skipif: requires a pg_dump on PATH whose major matches NEON_SERVER_MAJOR (17).
    Network-free — --check resolves the URL and tooling but never connects.
    """
    pg17 = _find_pg_dump(17)
    if not pg17:
        pytest.skip("no postgresql-client-17 pg_dump on PATH for the preflight contract test")
    env = dict(os.environ)
    env["NEON_SERVER_MAJOR"] = "17"
    env["PATH"] = str(Path(pg17).parent) + os.pathsep + env.get("PATH", "")
    cp = subprocess.run(
        ["bash", str(BACKUP_SH), "--check", "--url", _DUMMY_URL],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert cp.returncode == 0, (
        f"backup_neon.sh --check should exit 0 with a pg17 client.\n"
        f"STDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
    )
    # It must report that tooling + URL were resolved (preflight contract).
    assert "preflight OK" in cp.stderr, "--check must report a successful preflight"


def test_backup_script_hint_text_says_17():
    """The script's human-facing hints must say PostgreSQL 17, not the stale 16.

    RED today: the `die`/header text said "PostgreSQL 16 client". GREEN: updated
    to 17 (and the install hints point at postgresql-client-17 / postgresql@17).
    """
    txt = BACKUP_SH.read_text()
    assert "PostgreSQL 16 client" not in txt, "stale 'PostgreSQL 16 client' hint must be updated to 17"
    assert "postgresql@16" not in txt, "stale macOS brew hint (postgresql@16) must be updated to 17"
    # Positive lock: the 17 hints are present.
    assert "postgresql-client-17" in txt and "postgresql@17" in txt, (
        "backup_neon.sh should hint at the v17 client (postgresql-client-17 / postgresql@17)"
    )
