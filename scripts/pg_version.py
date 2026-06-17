#!/usr/bin/env python3
"""
pg_version.py — single source of truth for the pg_dump major-version preflight.

WHY (#7 / Problem 2): The Neon prod backup failed daily because CI installed
postgresql-client **16** while the Neon server is **17.10**, so `pg_dump` aborted
at dump time with "server version mismatch". backup_neon.sh's `--check` preflight
only verified that a pg_dump binary EXISTS — it never compared the client's major
against the server's. This module closes that gap with a pure, deterministic
version-compare that needs NO database connection and NO secret, so it is fully
unit-testable (pg16 → Red, pg17 → Green) and can run in CI.

It is the SHARED engine: backup_neon.sh shells out to this (`--check-major`) so the
bash gate and the Python tests can never drift. The real dump-time server-version
check inside pg_dump stays as the backstop — this just surfaces the mismatch
EARLIER (in `--check`) and WITHOUT needing the live server.

CLI:
  pg_version.py --extract "pg_dump (PostgreSQL) 17.9 (Homebrew)"   -> prints "17"
  pg_version.py --pg-dump /path/to/pg_dump                         -> prints major, e.g. "17"
  pg_version.py --check-major 17 [--pg-dump PATH]                  -> exit 0 if major==17 else 1

Read-only: never connects to a database.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys


class PgVersionError(RuntimeError):
    """Raised when a pg_dump version cannot be determined."""


def parse_major(version_output: str) -> int:
    """Extract the MAJOR version from `pg_dump --version` output.

    Handles every vendor form we see in the wild:
      "pg_dump (PostgreSQL) 17.9 (Homebrew)"                 -> 17
      "pg_dump (PostgreSQL) 16.14 (Ubuntu 16.14-1.pgdg24...)" -> 16
      "pg_dump (PostgreSQL) 17.10"                            -> 17
      "pg_dump (PostgreSQL) 10.23"                            -> 10  (pre-v10 had x.y.z; v10+ is major.minor)

    The major is the first integer in the first "<num>.<num>" (or bare "<num>")
    token that follows the "(PostgreSQL)" marker.
    """
    if not version_output:
        raise PgVersionError("empty pg_dump --version output")
    # Anchor on the product marker so a stray number elsewhere can't fool us.
    m = re.search(r"PostgreSQL\)\s+(\d+)(?:\.\d+)*", version_output)
    if not m:
        # Fallback: first standalone version-looking token anywhere.
        m = re.search(r"\b(\d+)\.\d+", version_output)
    if not m:
        raise PgVersionError(f"could not parse a version from: {version_output!r}")
    return int(m.group(1))


def pg_dump_major(pg_dump: str = "pg_dump") -> int:
    """Run `<pg_dump> --version` and return its major version as an int."""
    exe = shutil.which(pg_dump) or pg_dump
    try:
        out = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except FileNotFoundError as e:
        raise PgVersionError(f"pg_dump not found: {pg_dump}") from e
    except subprocess.CalledProcessError as e:
        raise PgVersionError(f"`{exe} --version` failed: {e.stderr or e}") from e
    except subprocess.TimeoutExpired as e:
        raise PgVersionError(f"`{exe} --version` timed out") from e
    return parse_major(out)


def majors_match(pg_dump: str, expected_major: int) -> bool:
    """True iff the pg_dump binary's major equals the expected server major."""
    return pg_dump_major(pg_dump) == int(expected_major)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="pg_dump major-version preflight helper")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--extract", metavar="VERSION_STRING",
                   help="parse the major from a `pg_dump --version` STRING and print it")
    g.add_argument("--print-major", action="store_true",
                   help="run pg_dump --version and print its major")
    g.add_argument("--check-major", type=int, metavar="EXPECTED",
                   help="exit 0 if pg_dump major == EXPECTED, else exit 1 (loud message)")
    p.add_argument("--pg-dump", default="pg_dump",
                   help="pg_dump binary to inspect (default: first on PATH)")
    args = p.parse_args(argv)

    try:
        if args.extract is not None:
            print(parse_major(args.extract))
            return 0
        if args.print_major:
            print(pg_dump_major(args.pg_dump))
            return 0
        # --check-major
        actual = pg_dump_major(args.pg_dump)
        if actual == args.check_major:
            return 0
        sys.stderr.write(
            f"pg_dump major {actual} != expected server major {args.check_major}. "
            f"pg_dump CANNOT dump a newer server (it aborts with 'server version mismatch'). "
            f"Install the matching client:\n"
            f"  Debian/Ubuntu (PGDG): sudo apt-get install -y postgresql-client-{args.check_major}\n"
            f"  macOS:                brew install postgresql@{args.check_major}\n"
        )
        return 1
    except PgVersionError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
