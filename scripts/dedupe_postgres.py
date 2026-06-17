#!/usr/bin/env python3
"""Cross-source duplicate detection for the daily pipeline (PROBLEM 1 — #5).

WHY THIS EXISTS
---------------
The "Clean duplicate listings" step in .github/workflows/daily-scrape.yml used to
inline a weak dedupe (lowercase + strip 'london/uk', 0.70 SequenceMatcher ratio,
transitive union-find, no same-source skip). On the real DB that proposed deleting
3095/7831 (39.5%) of active listings — chaining whole postcode districts into one
cluster — and the _safe_delete guard correctly aborted (exit 1), which skipped the
retrain. Research (docs/research_dedupe.md) proved the true cross-source signal is
only ~53 rows (~0.7%): the 39.5% was an over-clustering BUG, not real duplication.

THE FIX
-------
Key on the STRUCTURAL `address_fingerprint` (property_scraper/services/fingerprint),
not fuzzy strings. fingerprint parses flat/unit + street number + street + postcode
and has a vague-address fallback (source+property_id) so distinct flats on a vague
"Abbey Road, NW8" can't collide. Identity rule:

  - group rows by (fingerprint, bedrooms);
  - inside a group, keep only members whose price is within ±5% of the group's
    median (sanity gate against a stray mismatch);
  - a group is a DUPLICATE set ONLY if it spans >=2 distinct sources
    (cross-source overlap — same-source relists are dedupe_same_source.py's job);
  - keep the best record per group by SOURCE_PRIORITY then has-sqft then id;
  - return the rest as ids-to-delete.

NO union-find, NO transitive chaining, NO same-source deletions. This is a pure,
DB-agnostic function: the caller supplies rows (list of dicts) and applies the
returned ids through scripts/_safe_delete.guarded_delete (which keeps the 10% guard
and writes a CSV backup).
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from statistics import median
from typing import Iterable

# Make the fingerprint service importable whether called from repo root, scripts/,
# or the workflow heredoc.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from property_scraper.services.fingerprint import generate_fingerprint

# ONE source-priority constant for the whole codebase: import it, never re-derive it.
# Higher = preferred as the canonical record to KEEP (agents have better sqft than the
# rightmove aggregator). dedupe_cross_source defines it as a plain module-level dict and
# only runs its CLI under __main__, so importing the constant has no side effects. A local
# copy here was the exact divergence risk (#5/#10) that let the inline dedupe drift.
from dedupe_cross_source import SOURCE_PRIORITY

# A group's members must agree on price within this fraction of the group median to
# be treated as the same property (guards against a stray fingerprint collision).
PRICE_TOLERANCE = 0.05


def _fingerprint(row: dict) -> str:
    return generate_fingerprint(
        row.get("address") or "",
        row.get("postcode") or "",
        source=row.get("source"),
        property_id=str(row.get("property_id")) if row.get("property_id") is not None else None,
    )


def _has_sqft(row: dict) -> int:
    return 1 if (row.get("size_sqft") or 0) > 0 else 0


def group_cross_source_clusters(rows: Iterable[dict]) -> list[list[dict]]:
    """Return the cross-source duplicate clusters (each a list of >=2 rows).

    A cluster is a set of rows sharing (fingerprint, bedrooms), agreeing on price
    within tolerance, and spanning >=2 distinct sources. Same-source-only groups
    are NOT returned (that is dedupe_same_source.py's responsibility).
    """
    by_key: dict[tuple[str, object], list[dict]] = defaultdict(list)
    for row in rows:
        fp = _fingerprint(row)
        # Skip rows we can't key (no usable fingerprint). Defensive — fingerprint
        # always returns a hash, but guard against empty just in case.
        if not fp:
            continue
        by_key[(fp, row.get("bedrooms"))].append(row)

    clusters: list[list[dict]] = []
    for members in by_key.values():
        if len(members) < 2:
            continue

        # Price sanity gate: keep only members within ±5% of the group median price.
        prices = [m["price_pcm"] for m in members if (m.get("price_pcm") or 0) > 0]
        if prices:
            med = median(prices)
            members = [
                m
                for m in members
                if (m.get("price_pcm") or 0) > 0
                and abs(m["price_pcm"] - med) <= PRICE_TOLERANCE * med
            ]
        if len(members) < 2:
            continue

        # Cross-source only: the surviving members must span >=2 distinct sources.
        if len({m["source"] for m in members}) < 2:
            continue

        clusters.append(members)

    return clusters


def find_cross_source_duplicate_ids(rows: Iterable[dict]) -> list[int]:
    """Return the ids of rows to DELETE as cross-source duplicates.

    For each cross-source cluster, keep the best record (highest SOURCE_PRIORITY,
    then has-sqft, then lowest id) and mark the rest for deletion. The returned
    ids should be passed through scripts/_safe_delete.guarded_delete by the caller.
    """
    rows = list(rows)
    to_delete: list[int] = []

    for cluster in group_cross_source_clusters(rows):
        # Best record to KEEP: highest priority, then has sqft, then lowest id.
        best = max(
            cluster,
            key=lambda r: (
                SOURCE_PRIORITY.get(r["source"], 0),
                _has_sqft(r),
                -r["id"],  # tie-break: prefer the smaller id
            ),
        )
        for r in cluster:
            if r["id"] != best["id"]:
                to_delete.append(r["id"])

    return sorted(to_delete)
