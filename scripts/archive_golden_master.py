#!/usr/bin/env python3
"""
archive_golden_master.py — immutable golden-master archive (task #38, dataeng half).

Writes ONE timestamped, integrity-verified, read-only copy of a dataset OUT of the
destructive path (output/, where the daily pipeline mark-inactive/dedupe/sync run)
into a separate golden_master/ tree. The golden master is the last-known-good
complete dataset; nothing in the pipeline ever writes or deletes it.

Why OUT of output/: the daily pipeline's backups live in output/backups/ and are
subject to _cleanup_old_backups() (age-based purge). The golden master must survive
that — it lives in a sibling dir the pipeline never touches, and is chmod'd read-only.

Usage:
  python3 scripts/archive_golden_master.py                          # archive output/rentals.db
  python3 scripts/archive_golden_master.py --src output/rentals_merged.db --label merged
  python3 scripts/archive_golden_master.py --list                   # list existing golden masters

Companion: security owns the cloud mirror (BACKUP_STRATEGY.md references this path).
"""
import argparse
import hashlib
import os
import shutil
import sqlite3
import stat
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = PROJECT_ROOT / "golden_master"          # sibling of output/, NOT inside it
DEFAULT_SRC = PROJECT_ROOT / "output" / "rentals.db"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def integrity_ok(path: Path) -> bool:
    try:
        conn = sqlite3.connect(path)
        ok = conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        conn.close()
        return ok
    except Exception:
        return False


def counts(path: Path) -> dict:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    out = {}
    for t in ("listings", "price_history"):
        try:
            out[t] = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            out[t] = None
    try:
        out["active"] = cur.execute("SELECT SUM(is_active) FROM listings").fetchone()[0]
    except Exception:
        out["active"] = None
    conn.close()
    return out


def archive(src: Path, label: str) -> Path:
    if not src.exists():
        sys.exit(f"ERROR: source not found: {src}")
    # 1. Verify the SOURCE is intact before we trust it as a golden master.
    if not integrity_ok(src):
        sys.exit(f"ERROR: source failed integrity_check, refusing to archive a corrupt master: {src}")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = GOLDEN_DIR / f"rentals_golden_{label}_{ts}.db"
    shutil.copy2(src, dest)
    # 2. Verify the COPY is also intact + byte-identical (no silent corruption on copy).
    if not integrity_ok(dest):
        dest.unlink()
        sys.exit("ERROR: copied golden master failed integrity_check — removed, aborting")
    src_hash, dest_hash = sha256(src), sha256(dest)
    if src_hash != dest_hash:
        dest.unlink()
        sys.exit("ERROR: golden master hash != source hash — removed, aborting")
    # 3. Write a sidecar manifest (counts + hash + provenance).
    c = counts(dest)
    manifest = dest.with_suffix(".db.manifest.txt")
    manifest.write_text(
        f"golden_master\nsource={src}\narchived_at={datetime.now().isoformat()}\n"
        f"label={label}\nsha256={dest_hash}\n"
        f"listings={c['listings']}\nactive={c['active']}\nprice_history={c['price_history']}\n"
    )
    # 4. Make the golden master + manifest READ-ONLY (defense vs accidental overwrite).
    for p in (dest, manifest):
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0444
    print(f"[golden] archived {src.name} -> {dest}")
    print(f"[golden]   sha256={dest_hash}")
    print(f"[golden]   listings={c['listings']} active={c['active']} price_history={c['price_history']}")
    print(f"[golden]   read-only (0444); manifest: {manifest.name}")
    print(f"[golden]   NOTE: lives OUTSIDE output/ — pipeline backups-cleanup never touches it.")
    return dest


def list_masters():
    if not GOLDEN_DIR.exists():
        print("(no golden_master/ dir yet)")
        return
    masters = sorted(GOLDEN_DIR.glob("rentals_golden_*.db"))
    if not masters:
        print("(no golden masters archived yet)")
        return
    print(f"Golden masters in {GOLDEN_DIR}:")
    for m in masters:
        mf = m.with_suffix(".db.manifest.txt")
        info = mf.read_text().strip().replace("\n", " | ") if mf.exists() else "(no manifest)"
        ro = "ro" if not (m.stat().st_mode & stat.S_IWUSR) else "RW(!)"
        print(f"  [{ro}] {m.name}  {info}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="DB to archive as golden master")
    ap.add_argument("--label", default="canonical", help="label (e.g. canonical, merged)")
    ap.add_argument("--list", action="store_true", help="list existing golden masters")
    args = ap.parse_args()
    if args.list:
        list_masters()
        return 0
    archive(Path(args.src), args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
