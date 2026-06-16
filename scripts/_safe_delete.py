"""
_safe_delete.py — reusable safety wrapper for destructive row deletions (#37).

Root-cause prevention: the prod-sync TRUNCATE destroyed data. Every DELETE path
(dedupe scripts, the CI dedupe step, etc.) should go through ONE guarded helper so
a bug (over-aggressive clustering, bad id list) can't silently wipe a large slice
of data, and any deletion is recoverable.

guarded_delete() does, in order:
  1. DELTA GUARD — abort if the deletion exceeds `max_fraction` of the table's current
     rows (default 10%). A correct dedupe retires a small tail; a >10% delete means a
     bug. Raises SafeDeleteAborted instead of deleting.
  2. BACKUP — dump the to-be-deleted rows (full columns) to a timestamped CSV under
     output/deleted_backups/ BEFORE deleting, so over-deletion is recoverable.
  3. DELETE — only then run the caller's delete (via a callback), inside the caller's txn.

Works for sqlite3 and psycopg2 cursors (paramstyle differs, so the caller supplies
the delete callback; this helper handles the count, guard, and backup).
"""
from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Callable, Sequence


class SafeDeleteAborted(RuntimeError):
    """Raised when a deletion exceeds the safety threshold — nothing was deleted."""


def _backup_dir(project_root: Path) -> Path:
    d = project_root / "output" / "deleted_backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def guarded_delete(
    cursor,
    table: str,
    ids: Sequence,
    *,
    total_rows: int,
    do_delete: Callable[[Sequence], None],
    project_root: Path | None = None,
    max_fraction: float = 0.10,
    label: str = "",
    select_sql: str | None = None,
) -> str:
    """Guard + back up + perform a row deletion.

    Args:
        cursor: an OPEN db cursor (sqlite3 or psycopg2) used to read the doomed rows.
        table: table name (for the backup filename + the default SELECT).
        ids: the primary-key ids about to be deleted.
        total_rows: current row count of `table` (for the delta guard).
        do_delete: callback that actually deletes `ids` (caller owns paramstyle/txn).
        project_root: repo root (defaults to this file's parent.parent).
        max_fraction: abort if len(ids)/total_rows exceeds this (default 0.10 = 10%).
        label: optional context string for logs.
        select_sql: optional SELECT to fetch the backup rows; must accept the id list.
                    Defaults to "SELECT * FROM {table} WHERE id IN (...)".

    Returns the backup file path. Raises SafeDeleteAborted if the guard trips.
    """
    n = len(ids)
    if n == 0:
        return ""

    if total_rows > 0:
        frac = n / total_rows
        if frac > max_fraction:
            raise SafeDeleteAborted(
                f"ABORTED {label or table}: would delete {n}/{total_rows} "
                f"({frac*100:.1f}%) rows — exceeds {max_fraction*100:.0f}% safety threshold. "
                f"Likely a bug. Nothing deleted; investigate before re-running."
            )

    root = project_root or Path(__file__).resolve().parent.parent
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = _backup_dir(root) / f"{table}_deleted_{ts}.csv"

    # Read the doomed rows for the backup. Detect paramstyle from the cursor module.
    paramstyle_qmark = "sqlite3" in type(cursor).__module__
    placeholder = "?" if paramstyle_qmark else "%s"
    ph = ",".join([placeholder] * n)
    sql = select_sql or f"SELECT * FROM {table} WHERE id IN ({ph})"
    cursor.execute(sql, list(ids))
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    with open(backup_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    # Perform the actual delete via the caller's callback (owns txn + paramstyle).
    do_delete(ids)

    print(
        f"[safe-delete] {label or table}: deleted {n}/{total_rows} "
        f"({(n/total_rows*100) if total_rows else 0:.1f}%); backup -> {backup_path}"
    )
    return str(backup_path)
