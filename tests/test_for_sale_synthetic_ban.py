"""test_for_sale_synthetic_ban.py — TDD contract for the SYNTHETIC-BAN guard (Directive A).

WHAT THIS FILE PINS (the for-sale vertical, correctness-critical guard)
----------------------------------------------------------------------
Production sale training must NEVER silently train on the committed synthetic sample, and
must NEVER ship a model from a too-thin REAL crawl. The synthetic path is opt-in,
unit-test-only, behind an explicit `allow_synthetic` keyword threaded
load_sale_rows -> run_sale_retrain:

  * T-A1 prod-mode (no allow_synthetic) + absent DB -> load_sale_rows RAISES loudly.
  * T-A2 prod-mode run_sale_retrain (write=False) + absent DB -> RAISES at load.
  * T-A3 allow_synthetic=True on the committed sample -> trains (returns the sample rows).
  * T-A4 a too-thin REAL sales.db (< MIN_REAL_SALE_ROWS usable rows) -> RAISES loudly,
         so a near-empty crawl can never ship a model.

These are REAL refusals (no mocks of the unit under test). The synthetic sample stays a
test-only fixture; production (the GHA workflow + the module __main__) never passes the
flag, so it can only train on a real, non-trivial sales.db or hard-fail.

ZERO RENTAL REGRESSION: nothing here imports or mutates the rental model chain.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import for_sale.sale_retrain as rt
from for_sale import sale_data, sale_price_model

pytestmark = pytest.mark.for_sale


# ── Empty-frame guard (AMENDMENT) ─────────────────────────────────────────────────────────
def test_build_features_empty_rows_raises_clear_runtimeerror():
    """build_features([]) on a 0-row frame must raise a CLEAR RuntimeError naming empty input,
    NOT the cryptic 'numpy.float64 object has no attribute fillna' AttributeError that the
    median-fill path produced before. The MIN_REAL_SALE_ROWS floor fires first on the prod
    path, but train() can be handed [] from other call paths."""
    with pytest.raises(RuntimeError, match="no rows|empty|0 rows"):
        sale_price_model.build_features([])


def test_train_empty_rows_raises_clear_runtimeerror():
    """train([]) must surface the same CLEAR RuntimeError (via build_features) rather than the
    cryptic fillna AttributeError, so an accidental empty-row training fails legibly."""
    with pytest.raises(RuntimeError, match="no rows|empty|0 rows"):
        sale_price_model.train([])


# ── T-A1 ────────────────────────────────────────────────────────────────────────────────
def test_prod_mode_synthetic_raises(tmp_path):
    """load_sale_rows with an ABSENT db_path and NO allow_synthetic must REFUSE the
    synthetic fallback loudly (UNIT-TEST-ONLY message), never silently load the sample."""
    absent = tmp_path / "nope.db"
    assert not absent.exists()
    with pytest.raises(RuntimeError, match="UNIT-TEST-ONLY"):
        rt.load_sale_rows(db_path=absent, sample_path=rt.DEFAULT_SAMPLE)


# ── T-A2 ────────────────────────────────────────────────────────────────────────────────
def test_run_sale_retrain_prod_mode_synthetic_raises(tmp_path):
    """run_sale_retrain in prod mode (no allow_synthetic) with an absent DB must RAISE at the
    load step (write=False so even a regressed guard writes no artifacts)."""
    absent = tmp_path / "nope.db"
    with pytest.raises(RuntimeError, match="UNIT-TEST-ONLY"):
        rt.run_sale_retrain(db_path=absent, write=False)


# ── T-A3 ────────────────────────────────────────────────────────────────────────────────
def test_allow_synthetic_opt_in_succeeds(tmp_path):
    """allow_synthetic=True is the explicit unit-test opt-in: with an absent DB it loads the
    committed synthetic sample and returns plausible sale rows (every row has asking_price)."""
    absent = tmp_path / "nope.db"
    rows = rt.load_sale_rows(db_path=absent, allow_synthetic=True)
    assert isinstance(rows, list) and len(rows) > 0
    assert all("asking_price" in r for r in rows), "sample rows must carry the asking_price label"


# ── T-A4 ────────────────────────────────────────────────────────────────────────────────
def test_thin_real_db_raises(tmp_path):
    """A REAL sales.db with fewer than MIN_REAL_SALE_ROWS usable rows must RAISE loudly: a
    too-thin crawl must NEVER ship a model. The DB branch is taken (db exists), so neither
    allow_synthetic nor the synthetic fallback is involved."""
    thin_db = tmp_path / "sales.db"
    conn = sqlite3.connect(str(thin_db))
    try:
        sale_data.create_schema(conn)
        # One single plausible, active, priced sale row — far below MIN_REAL_SALE_ROWS.
        sale_data.upsert_sale_listing(
            conn,
            {
                "source": "rightmove",
                "property_id": "thin-1",
                "asking_price": 1_500_000,
                "address": "1 A Street, London, SW3 4TX",
                "postcode": "SW3 4TX",
                "bedrooms": 2,
                "bathrooms": 2,
                "property_type": "Flat",
                "size_sqft": 900,
                "is_active": 1,
            },
        )
    finally:
        conn.close()

    assert thin_db.exists()
    assert rt.MIN_REAL_SALE_ROWS > 1, "the floor must exceed the single thin row for this test"
    with pytest.raises(RuntimeError, match="too-thin|usable rows"):
        rt.load_sale_rows(db_path=thin_db)
