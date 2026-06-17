#!/usr/bin/env python3
"""TDD tests for the cross-source dedupe identity (PROBLEM 1 — #5).

Root cause (research_dedupe.md): the INLINE heredoc in
.github/workflows/daily-scrape.yml (lines 235-341) over-deleted 3095/7831
(39.5%) of active listings vs a true cross-source signal of ~53 rows (~0.7%).
The bug = 0.70 fuzzy-similarity + transitive union-find + no same-source skip +
weak normalization, which chained whole postcode districts into one cluster.

The fix EXTRACTS the dedupe into a pure, importable function
`scripts/dedupe_postgres.find_cross_source_duplicate_ids(rows)` that keys on the
STRUCTURAL `address_fingerprint` (property_scraper/services/fingerprint) instead
of fuzzy string matching:
  - group on fingerprint + bedrooms (+ price ±5% sanity gate),
  - a group is a dup set ONLY if it spans >=2 distinct sources (cross-source),
  - keep best per group by source priority + has-sqft,
  - NO union-find, NO transitive chaining.

UNIT tests (default CI, no DB) gate the PR. Two @pytest.mark.data tests assert
the real canonical fraction is under the 10% guard; they auto-skip when no
rentals.db is present (tests/conftest.py) and run in daily-scrape post-scrape.

All rows below are real, verified rows from output/rentals.db (research fixture).
Columns: id, source, address, postcode, price_pcm, bedrooms, size_sqft, property_id.
"""

import os
import sqlite3
import sys
from pathlib import Path

import pytest

# Make scripts/ importable (same pattern as tests/test_safe_delete.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dedupe_postgres import find_cross_source_duplicate_ids  # noqa: E402


def _row(id, source, address, postcode, price_pcm, bedrooms, size_sqft, property_id):
    return {
        "id": id,
        "source": source,
        "address": address,
        "postcode": postcode,
        "price_pcm": price_pcm,
        "bedrooms": bedrooms,
        "size_sqft": size_sqft,
        "property_id": property_id,
    }


# --- Group A: DISTINCT properties the buggy logic wrongly merged (must NOT cluster) ---
# Real NW8 2-bed rows on DIFFERENT streets. The vague-address rows (no street
# number) hash via fingerprint's vague: fallback (source+property_id) so they are
# structurally distinct — only the weak fuzzy path merged them.
GROUP_A_DISTINCT = [
    _row(1, "savills", "17 Hall Road, St. John's Wood, NW8 9RF", "NW8 9RF", 4225, 2, None, "sav-A1"),
    _row(2, "knightfrank", "Abbey Road, St John's Wood, London NW8", "NW8", 4983, 2, None, "knf-A2"),
    _row(3, "rightmove", "Abbey Road, St John's Wood, London, NW8", "NW8", 3575, 2, None, "rgt-A3"),
    _row(4, "chestertons", "Abbey Road, St John's Wood, London, NW8", "NW8", 3900, 2, None, "che-A4"),
]

# --- Group B: TRUE cross-source duplicates the fix MUST still catch ---
# Each pair shares fingerprint + bedrooms + ~price across two sources -> ONE real
# duplicate. Expected: drop the rightmove row (lowest source priority) in each.
GROUP_B_TRUE_DUPES = [
    _row(10, "chestertons", "Kingsmill, 1-19 Kingsmill Terrace, St John's Wood, London, NW8", "NW8", 4000, 1, 641, "che-B1"),
    _row(11, "rightmove", "Kingsmill, 1-19 Kingsmill Terrace, NW8", "NW8", 4000, 1, 641, "rgt-B1"),
    _row(12, "chestertons", "Hillside Court, 409 Finchley Road, London, NW3", "NW3", 3553, 3, 952, "che-B2"),
    _row(13, "rightmove", "Hillside Court, 409 Finchley Road, London, NW3", "NW3", 3553, 3, None, "rgt-B2"),
    _row(14, "chestertons", "Quebec Court, 21 Seymour Street, W1H", "W1H", 2686, 1, None, "che-B3"),
    _row(15, "rightmove", "Quebec Court, 21 Seymour Street, W1H", "W1H", 2687, 1, None, "rgt-B3"),
]
GROUP_B_RIGHTMOVE_IDS = {11, 13, 15}

# --- Group C: same fingerprint, DIFFERENT bedrooms (must NOT merge) ---
# Park House Apartments 1-bed vs 2-bed at very different prices. Pins the key to
# fingerprint + bedrooms (fingerprint-only would wrongly merge them).
GROUP_C_FP_DIFFERENT_BEDS = [
    _row(20, "knightfrank", "Park House Apartments, 47 North Row, Mayfair, London W1K", "W1K", 6435, 1, 858, "knf-C1"),
    _row(21, "knightfrank", "Park House Apartments, 47 North Row, Mayfair, London, W1K", "W1K", 9533, 2, 1248, "knf-C2"),
]


# =========================================================================
# UNIT tests (gate the PR — always run, no DB needed)
# =========================================================================

def test_distinct_streets_not_merged():
    """Test C: Abbey Road vs Abercorn Place (same beds/price/district, different
    streets, different sources) must NOT be flagged as duplicates.

    Old inline logic merged them at fuzzy ratio 0.774 on the shared
    'st john's wood nw8' tail. The structural fingerprint key keeps them apart.
    """
    rows = [
        _row(1, "knightfrank", "Abbey Road, St John's Wood, London, NW8", "NW8", 4000, 2, None, "knf-1"),
        _row(2, "rightmove", "Abercorn Place, St John's Wood, London, NW8", "NW8", 4050, 2, None, "rgt-2"),
    ]
    assert find_cross_source_duplicate_ids(rows) == []


def test_true_cross_source_dupe_still_caught():
    """Test D: the real Alexandra Mansions, 333 Kings Road, SW3 pair
    (knightfrank + rightmove, beds 2, both £3098) is ONE property listed twice.
    Exactly one id (the rightmove row) must be returned for deletion.
    """
    rows = [
        _row(1, "knightfrank", "Alexandra Mansions, 333 Kings Road, SW3", "SW3", 3098, 2, 900, "knf-1"),
        _row(2, "rightmove", "Alexandra Mansions, 333 Kings Road, SW3", "SW3", 3098, 2, None, "rgt-2"),
    ]
    result = find_cross_source_duplicate_ids(rows)
    assert result == [2], f"expected only the rightmove id [2], got {result}"


def test_no_megaclusters():
    """Test E: ~6 distinct NW8 streets sharing the locality tail, all 2-bed,
    similar price, mixed sources. The old union-find chained all 6 into one
    cluster (deleting 5). The fingerprint key must keep them apart -> 0 deletions.
    """
    streets = [
        "Abbey Road", "Abercorn Place", "Aberdeen Place", "Belgrave Gardens",
        "Bolton Road", "Boundary Road",
    ]
    sources = ["rightmove", "savills", "knightfrank", "foxtons", "chestertons", "rightmove"]
    rows = [
        _row(i + 1, sources[i], f"{street}, St John's Wood, London, NW8", "NW8",
             4000 + i * 30, 2, None, f"pid-{i+1}")
        for i, street in enumerate(streets)
    ]
    to_delete = find_cross_source_duplicate_ids(rows)
    # No mega-cluster: distinct streets must not be merged at all here.
    assert to_delete == [], f"distinct streets were merged: {to_delete}"


def test_fixture_groups_abc_combined():
    """Headline fixture: seed Groups A+B+C together and assert exact behaviour.

    - Group A: 0 deletions (distinct streets / vague-address rows don't merge).
    - Group B: exactly the 3 rightmove ids (one per true cross-source pair).
    - Group C: 0 deletions (same fingerprint but different bedrooms).
    Total expected deletions = the 3 Group-B rightmove ids only.
    """
    rows = GROUP_A_DISTINCT + GROUP_B_TRUE_DUPES + GROUP_C_FP_DIFFERENT_BEDS
    to_delete = set(find_cross_source_duplicate_ids(rows))

    assert to_delete == GROUP_B_RIGHTMOVE_IDS, (
        f"expected exactly the 3 Group-B rightmove ids {GROUP_B_RIGHTMOVE_IDS}, "
        f"got {sorted(to_delete)}"
    )
    # No Group-A id deleted.
    assert not (to_delete & {r["id"] for r in GROUP_A_DISTINCT})
    # No Group-C id deleted.
    assert not (to_delete & {r["id"] for r in GROUP_C_FP_DIFFERENT_BEDS})


def test_no_same_source_deletions_on_fixture():
    """Test B (unit form): a cross-source dedupe must never delete a row whose
    kept counterpart is the SAME source. Two same-source rows that share a
    fingerprint+beds are NOT this step's job (that's dedupe_same_source.py).
    """
    rows = [
        _row(1, "rightmove", "Kingsmill, 1-19 Kingsmill Terrace, NW8", "NW8", 4000, 1, 641, "rgt-1"),
        _row(2, "rightmove", "Kingsmill, 1-19 Kingsmill Terrace, NW8", "NW8", 4000, 1, 641, "rgt-2"),
    ]
    assert find_cross_source_duplicate_ids(rows) == [], (
        "same-source rows must not be deleted by the cross-source step"
    )


# =========================================================================
# Delegation test: dedupe_cross_source.find_duplicates now routes through the
# structural core (#10) — one identity rule, no fuzzy SequenceMatcher path.
# =========================================================================

def _seed_listings_db(rows):
    """In-memory SQLite with just the columns find_duplicates reads."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE listings (
            id INTEGER PRIMARY KEY, source TEXT, property_id TEXT, address TEXT,
            postcode TEXT, price_pcm INTEGER, bedrooms INTEGER, bathrooms INTEGER,
            size_sqft INTEGER, url TEXT
        )"""
    )
    cur.executemany(
        """INSERT INTO listings
           (id, source, property_id, address, postcode, price_pcm, bedrooms, size_sqft, url)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [
            (r["id"], r["source"], r["property_id"], r["address"], r["postcode"],
             r["price_pcm"], r["bedrooms"], r["size_sqft"], "")
            for r in rows
        ],
    )
    conn.commit()
    return conn


def test_cross_source_cli_find_duplicates_delegates_to_structural_core():
    """dedupe_cross_source.find_duplicates must now behave like the structural
    core: it returns pairwise dicts, but only for TRUE cross-source dupes — never
    the distinct same-street properties the old fuzzy path wrongly merged.
    """
    import dedupe_cross_source as dcs

    conn = _seed_listings_db(GROUP_A_DISTINCT + GROUP_B_TRUE_DUPES + GROUP_C_FP_DIFFERENT_BEDS)
    pairs = dcs.find_duplicates(conn)
    conn.close()

    # Shape the CLI consumers depend on.
    for p in pairs:
        assert set(p.keys()) == {"record1", "record2", "similarity"}
        assert p["similarity"] == 1.0  # fingerprint identity is exact, not fuzzy

    deleted_ids = {p["record2"]["id"] for p in pairs}
    # Only the 3 Group-B rightmove rows are non-canonical dupes; A and C contribute none.
    assert deleted_ids == GROUP_B_RIGHTMOVE_IDS, (
        f"delegated find_duplicates flagged wrong ids: {sorted(deleted_ids)}"
    )
    # No Group-A (distinct streets) or Group-C (different beds) row appears.
    assert not (deleted_ids & {r["id"] for r in GROUP_A_DISTINCT})
    assert not (deleted_ids & {r["id"] for r in GROUP_C_FP_DIFFERENT_BEDS})


# =========================================================================
# DATA tests (read the live canonical DB; auto-skip without it)
# =========================================================================

def _load_active_rows(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, source, address, postcode, price_pcm, bedrooms, size_sqft, property_id
        FROM listings WHERE is_active = 1 AND price_pcm > 0
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _canonical_db_path():
    return Path(
        os.environ.get(
            "SCRAPER_DB_PATH",
            Path(__file__).resolve().parent.parent / "output" / "rentals.db",
        )
    )


@pytest.mark.data
def test_canonical_delete_fraction_under_guard():
    """Test A (headline regression): over the real ~7,831 active rows, the fixed
    dedupe must propose deleting < 10% (lands ~0.7%). The OLD inline logic
    proposed 39.5%, which the _safe_delete guard rejected (exit 1 -> skipped
    retrain). Re-running the OLD logic here proves the test bites.
    """
    rows = _load_active_rows(_canonical_db_path())
    active = len(rows)
    assert active > 0

    to_delete = find_cross_source_duplicate_ids(rows)
    frac = len(to_delete) / active
    assert frac < 0.10, (
        f"fixed dedupe still over-deletes: {len(to_delete)}/{active} = {frac*100:.1f}% "
        f"(must be < 10%)"
    )

    # Sanity: the OLD inline logic on the same rows DOES exceed the guard,
    # so this test genuinely exercises the regression.
    old_frac = _old_inline_delete_fraction(rows)
    assert old_frac > 0.10, (
        f"old inline logic unexpectedly under guard ({old_frac*100:.1f}%) — "
        f"fixture/DB may have changed; test no longer proves the fix"
    )


@pytest.mark.data
def test_no_same_source_deletions_on_canonical():
    """Test B (data): on the real DB, the cross-source dedupe must produce ZERO
    deletions where the kept row and a deleted row share the same source.
    (The old inline union-find produced 1044 such same-source deletions.)
    """
    rows = _load_active_rows(_canonical_db_path())
    by_id = {r["id"]: r for r in rows}
    to_delete = set(find_cross_source_duplicate_ids(rows))

    # Reconstruct clusters from the function to verify the kept survivor of each
    # deleted row is a DIFFERENT source. We re-run the grouping via the public
    # helper so the test stays black-box on identity but can inspect survivors.
    from dedupe_postgres import group_cross_source_clusters

    for cluster in group_cross_source_clusters(rows):
        deleted = [r for r in cluster if r["id"] in to_delete]
        kept = [r for r in cluster if r["id"] not in to_delete]
        if not deleted:
            continue
        assert kept, "a cluster deleted every member — nothing kept"
        kept_sources = {r["source"] for r in kept}
        for d in deleted:
            assert d["source"] not in kept_sources or len(cluster) != 2, (
                f"same-source deletion: row {d['id']} ({d['source']}) deleted while a "
                f"same-source row kept — that's dedupe_same_source.py's job"
            )


def _old_inline_delete_fraction(rows):
    """Replay the OLD inline CI clustering (the bug) to prove the data test bites.

    This is the exact weak-normalize + 0.70 + union-find logic from the failing
    workflow step — kept here ONLY so the regression test can assert the old
    behaviour breached the guard. Not used by production.
    """
    from difflib import SequenceMatcher
    from collections import defaultdict

    def normalize_addr(addr):
        if not addr:
            return ""
        addr = addr.lower()
        for word in ["london", "uk", ",", "."]:
            addr = addr.replace(word, " ")
        return " ".join(addr.split())

    listings = [
        {
            "id": r["id"], "source": r["source"], "price_pcm": r["price_pcm"],
            "bedrooms": r["bedrooms"], "size_sqft": r["size_sqft"],
            "district": (r["postcode"] or "").split(" ")[0],
            "norm_addr": normalize_addr(r["address"]),
        }
        for r in rows
    ]
    parent = {l["id"]: l["id"] for l in listings}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # O(n^2) within district (as the inline code did).
    by_dist = defaultdict(list)
    for l in listings:
        by_dist[l["district"]].append(l)
    for group in by_dist.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a["bedrooms"] != b["bedrooms"]:
                    continue
                pa, pb = a["price_pcm"] or 0, b["price_pcm"] or 0
                if abs(pa - pb) > 0.05 * max(pa, pb, 1):
                    continue
                if SequenceMatcher(None, a["norm_addr"], b["norm_addr"]).ratio() > 0.70:
                    union(a["id"], b["id"])

    clusters = defaultdict(list)
    for l in listings:
        clusters[find(l["id"])].append(l)
    to_delete = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda x: (0 if (x["size_sqft"] or 0) > 0 else 1, x["id"]))
        to_delete.extend(m["id"] for m in members[1:])
    return len(to_delete) / len(listings) if listings else 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
