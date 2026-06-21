"""test_for_sale_sale_model.py — TDD contract for the BASELINE SALE-PRICE MODEL
(for-sale vertical, increment #1 part 2 / Inc3-baseline).

ARCHITECTURE (read before changing the test)
--------------------------------------------
The for-sale vertical predicts a property's FAIR SALE VALUE from comparable
SCRAPED FOR-SALE ASKING listings — exactly as the rental v20 model predicts fair
rent from comparable rentals, but on the sale magnitude (£100k–£50M lump sums, not
£/month). This module is the for-sale analogue of rental_price_models_v20 but is a
SEPARATE module (for_sale/sale_price_model.py), trains a SEPARATE XGBoost model to a
SEPARATE artifact (output/sale_model.pkl), and predicts via a SEPARATE path.

It MIRRORS the rental feature-engineering PATTERNS (size / log_sqft / size_per_bed,
postcode→district boundary-anchored regex, district & prestige-street encodings,
beds / baths / bath_ratio, property-type one-hots) by RE-IMPLEMENTING them — it does
NOT import rental_price_models_v20 / canonical_predict (hard isolation guard below),
so a rental retrain can never perturb the sale model and vice-versa.

TARGET = log1p(asking_price). Magnitude is sale-scale: sale price-per-sqft (ppsf) in
prime London is ~£500–£3,000/sqft, vs rent ~£40–£100/sqft/YEAR — the bounds here are
sale-scale, and a rental-magnitude value is rejected as out-of-range.

LEAK-SAFETY: features are derived from the property only (size / location / type /
beds / baths). There is NO target-derived branch (no ppsf-threshold feature like the
rental is_social_housing leak that was removed). asking_price is the label, never a
feature.

CI-SAFETY: marker `for_sale` (registered under strict_markers). The model trains on a
COMMITTED deterministic for-sale sample (tests/fixtures/for_sale/sale_training_sample.json)
— no network, no live DB, deterministic seed — so it ALWAYS runs and gates the PR.
Kept on the anti-silent-skip allowlist.

ZERO RENTAL REGRESSION: nothing here imports or mutates rental_price_models_v20,
canonical_predict, the parity-gated xgboost.js, the rental model artifact, or the
rental `listings`/sale_listings tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.for_sale

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "for_sale" / "sale_training_sample.json"


@pytest.fixture(scope="module")
def model_mod():
    from for_sale import sale_price_model
    return sale_price_model


@pytest.fixture(scope="module")
def sample_rows():
    rows = json.loads(SAMPLE.read_text())
    assert isinstance(rows, list)
    return rows


# ── The committed deterministic for-sale training sample ──────────────────────────

def test_training_sample_exists_and_is_sale_magnitude():
    assert SAMPLE.exists(), (
        "committed for-sale training sample missing — it is the CI-safe, deterministic "
        "dataset the baseline sale model trains/validates against (no network)."
    )
    rows = json.loads(SAMPLE.read_text())
    assert isinstance(rows, list) and len(rows) >= 120, (
        "need enough rows for a meaningful train/test split + an R² assertion"
    )
    for r in rows:
        # SALE magnitude, not rent: every label is a lump sum in the London sale range.
        assert 100_000 <= r["asking_price"] <= 60_000_000
        # The features a sale comp carries (same attributes the rental item has).
        for k in ("postcode", "bedrooms", "bathrooms", "size_sqft", "property_type", "address"):
            assert k in r


# ── Feature engineering mirrors the rental PATTERNS (re-implemented, not imported) ─

def test_feature_engineering_builds_tenure_agnostic_features(model_mod, sample_rows):
    X, feature_cols = model_mod.build_features(sample_rows)
    assert len(X) == len(sample_rows)
    # The tenure-agnostic feature family the rental model also uses (mirrored).
    for f in ("log_sqft", "size_per_bed", "bath_ratio", "is_prime_postcode",
              "prestige_tier", "beds_squared"):
        assert f in feature_cols, f"expected mirrored feature {f}"
    # NO rent-specific / leaky columns.
    for banned in ("price_pcm", "ppsf", "asking_price", "is_social_housing",
                   "let_type", "furnished"):
        assert banned not in feature_cols, f"leaky/rent-only feature must be absent: {banned}"
    # All feature columns are numeric (model-ready, no object dtype).
    assert X[feature_cols].select_dtypes(include="number").shape[1] == len(feature_cols)


def test_district_regex_is_boundary_anchored(model_mod):
    """Mirror the rental FIX-1 boundary-anchored outward-code regex: a no-space
    postcode 'SW72ED' must yield district 'SW7' (NOT the greedy 'SW72E'); real
    sub-districts like 'SW1X' / 'W1K' are preserved; absent postcode → 'UNKNOWN'."""
    assert model_mod.postcode_to_district("SW72ED") == "SW7"
    assert model_mod.postcode_to_district("SW3 4TX") == "SW3"
    assert model_mod.postcode_to_district("SW1X") == "SW1X"
    assert model_mod.postcode_to_district("W1K") == "W1K"
    assert model_mod.postcode_to_district(None) == "UNKNOWN"
    assert model_mod.postcode_to_district("") == "UNKNOWN"


def test_prime_postcode_and_prestige_match_rental_semantics(model_mod):
    """is_prime_postcode fires on the prime districts; prestige_tier is 0 for an
    ordinary street and >0 for a known prestige street (mirrors rental encodings)."""
    X_prime, _ = model_mod.build_features([{
        "postcode": "SW3", "bedrooms": 2, "bathrooms": 2, "size_sqft": 900,
        "property_type": "Flat", "address": "Cadogan Square, London, SW3",
        "asking_price": 3_000_000,
    }])
    X_ord, _ = model_mod.build_features([{
        "postcode": "SE15", "bedrooms": 2, "bathrooms": 1, "size_sqft": 800,
        "property_type": "Flat", "address": "Some Ordinary Road, London, SE15",
        "asking_price": 600_000,
    }])
    assert int(X_prime["is_prime_postcode"].iloc[0]) == 1
    assert int(X_ord["is_prime_postcode"].iloc[0]) == 0
    assert X_prime["prestige_tier"].iloc[0] > 0   # Cadogan Square is a prestige street
    assert X_ord["prestige_tier"].iloc[0] == 0


# ── Leak-safety: target never becomes a feature; no target-derived branch ─────────

def test_no_target_leakage_in_features(model_mod, sample_rows):
    """asking_price (the label) must not appear as a feature, and no feature may be a
    deterministic function of it (the rental is_social_housing ppsf-branch leak class)."""
    X, feature_cols = model_mod.build_features(sample_rows)
    assert "asking_price" not in feature_cols
    # Perturb ONLY the label; features must be byte-identical (no target dependence).
    perturbed = [{**r, "asking_price": r["asking_price"] * 3} for r in sample_rows]
    X2, cols2 = model_mod.build_features(perturbed)
    assert cols2 == feature_cols
    assert np.allclose(X[feature_cols].to_numpy(dtype=float),
                       X2[feature_cols].to_numpy(dtype=float), equal_nan=True)


# ── Magnitude / ppsf bounds are SALE-scale, not rent-scale ────────────────────────

def test_sale_magnitude_bounds_reject_rent_scale(model_mod):
    assert model_mod.is_plausible_sale_price(875_000) is True
    assert model_mod.is_plausible_sale_price(40_000_000) is True
    assert model_mod.is_plausible_sale_price(3_500) is False     # a monthly rent
    assert model_mod.is_plausible_sale_price(0) is False
    assert model_mod.is_plausible_sale_price(None) is False


def test_sale_ppsf_bounds_are_prime_london_scale(model_mod):
    """Sale ppsf ~ £500–£3,000/sqft in prime London (vs rent ~£40–£100/sqft/yr). The
    training filter keeps in-range rows and drops absurd ones."""
    lo, hi = model_mod.SALE_PPSF_MIN, model_mod.SALE_PPSF_MAX
    assert 200 <= lo <= 800
    assert 2_000 <= hi <= 8_000
    # £900k for a 900 sqft flat = £1,000/sqft -> in range.
    assert model_mod.is_plausible_sale_ppsf(900_000 / 900) is True
    # £40/sqft is a RENT ppsf, not a sale ppsf.
    assert model_mod.is_plausible_sale_ppsf(40) is False


# ── Train → evaluate on a deterministic split (the baseline-model assertion) ──────

@pytest.fixture(scope="module")
def trained(model_mod, sample_rows):
    return model_mod.train(sample_rows, seed=42, test_size=0.25)


def test_train_returns_model_and_finite_metrics(trained):
    assert trained["model"] is not None
    m = trained["metrics"]
    for k in ("r2", "mae", "median_ape"):
        assert k in m and np.isfinite(m[k])
    assert m["mae"] > 0
    assert 0 <= m["median_ape"] < 1.0


def test_baseline_model_learns_signal(trained):
    """The whole point of a baseline: it must explain most of the variance on the
    held-out split. The committed sample is generated from a documented price formula
    with bounded noise, so a competent XGBoost baseline clears R² > 0.6."""
    assert trained["metrics"]["r2"] > 0.6, (
        f"baseline sale model under-fit: R²={trained['metrics']['r2']:.3f}"
    )


def test_predict_one_returns_finite_sale_price(model_mod, trained):
    """A single comparable property predicts a finite £ in the sane London sale range —
    the serving-shaped entry point (mirrors canonical_predict.predict_one)."""
    price = model_mod.predict_one(
        trained["model"], trained["feature_cols"],
        postcode="SW3", bedrooms=2, bathrooms=2, size_sqft=900,
        property_type="Flat", address="Cadogan Gardens, London, SW3",
    )
    assert np.isfinite(price)
    assert model_mod.is_plausible_sale_price(price), price
    # A bigger prime flat should not predict CHEAPER than a small one, all else equal
    # (sanity on monotonic size response — not an exact value).
    small = model_mod.predict_one(
        trained["model"], trained["feature_cols"],
        postcode="SW3", bedrooms=1, bathrooms=1, size_sqft=450,
        property_type="Flat", address="Cadogan Gardens, London, SW3",
    )
    assert price > small


def test_predict_handles_missing_postcode_without_crash(model_mod, trained):
    """A coordless / postcode-less comp must predict a finite price (the single-row
    object-dtype crash class the rental side hit) — district falls back to UNKNOWN."""
    price = model_mod.predict_one(
        trained["model"], trained["feature_cols"],
        postcode=None, bedrooms=2, bathrooms=2, size_sqft=850,
        property_type="Flat", address="Somewhere With No Postcode, London",
    )
    assert np.isfinite(price)
    assert model_mod.is_plausible_sale_price(price)


# ── Save / load round-trip to the SEPARATE sale artifact ──────────────────────────

def test_save_and_load_round_trip(model_mod, trained, tmp_path):
    """The model serializes to a SEPARATE sale artifact (NOT the rental pkl) and a
    reloaded model predicts identically."""
    art = tmp_path / "sale_model.pkl"
    feats = tmp_path / "sale_model_features.pkl"
    model_mod.save_model(trained["model"], trained["feature_cols"], art, feats)
    assert art.exists() and feats.exists()
    model2, cols2 = model_mod.load_model(art, feats)
    assert cols2 == trained["feature_cols"]
    p1 = model_mod.predict_one(trained["model"], trained["feature_cols"],
                               postcode="SW7", bedrooms=3, bathrooms=2, size_sqft=1400,
                               property_type="Flat", address="Onslow Square, London, SW7")
    p2 = model_mod.predict_one(model2, cols2,
                               postcode="SW7", bedrooms=3, bathrooms=2, size_sqft=1400,
                               property_type="Flat", address="Onslow Square, London, SW7")
    assert np.isclose(p1, p2)


# ── ISOLATION GUARDS — the sale model must not couple to the rental stack ─────────

def test_sale_model_does_not_import_rental_chain(model_mod):
    src = Path(model_mod.__file__).read_text()
    for banned in (
        "import rental_price_models_v20",
        "from rental_price_models_v20",
        "import canonical_predict",
        "from canonical_predict",
    ):
        assert banned not in src, f"sale model illegally couples to rental: {banned}"


def test_sale_model_artifact_path_is_separate_from_rental(model_mod):
    """The default artifact path must be the sale model, never the rental canonical pkl."""
    p = str(model_mod.DEFAULT_MODEL_PATH)
    assert "sale_model" in p
    assert "rental_model" not in p
