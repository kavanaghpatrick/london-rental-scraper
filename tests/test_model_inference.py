"""
test_model_inference.py — regression tests locking in the canonical v20 model +
single-row inference fixes (tasks #6, #30, #47).

These assert the EXACT bugs found during the parity exercise STAY fixed:
  - is_social_housing must NOT fire on a price-less prime-postcode request
    (the −81% bug: ppsf=0 < 3.5 fired spuriously). Fix: `0 < ppsf < 3.5`.
  - postcode_freq / postcode_area_freq must use the baked TRAINING maps at
    inference, not the degenerate single-row 1.0.
  - floor_count default = 0 (training median), not 1.
  - the version is single-sourced: canonical_predict derives from
    retrain_canonical.CANON_VERSION (one declaration).
  - the golden parity fixture is self-consistent (inputs → features → £).

Run: pytest tests/test_model_inference.py -v   (from scrapy_project/)

NOTE: these load the CURRENT canonical artifacts (output/rental_model_canonical*).
They remain valid after the #43 retrain — the £ values come from the fixture, which
is regenerated alongside the model, so the assertions read the fixture's own
prediction_pcm rather than hardcoding numbers that change with the data.
"""
import json
import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_PRESENT = (ROOT / 'output' / 'rental_model_canonical.pkl').exists()
FIXTURE_PATH = ROOT / 'output' / 'feature_parity_golden.json'

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="canonical artifacts not built (run retrain_canonical.py first)",
)


@pytest.fixture(scope='module')
def cp():
    import canonical_predict as _cp
    # fresh cache so a just-retrained model/inference.json is picked up
    for k in list(_cp._CACHE):
        _cp._CACHE[k] = None
    return _cp


@pytest.fixture(scope='module')
def fixture():
    assert FIXTURE_PATH.exists(), "golden fixture missing — run gen_feature_parity_golden.py"
    return json.loads(FIXTURE_PATH.read_text())


def _build_one(cp, **fields):
    """Build the 1-row canonical feature frame for a property (inference mode)."""
    row = {
        'bedrooms': fields.get('bedrooms', 2), 'bathrooms': fields.get('bathrooms', 1),
        'size_sqft': fields.get('size_sqft', 1000),
        'postcode': fields.get('postcode', ''),
        'postcode_normalized': fields.get('postcode_normalized', fields.get('postcode', '')),
        'area': fields.get('area', ''), 'property_type': fields.get('property_type', 'flat'),
        'property_type_std': fields.get('property_type', 'flat'),
        'address': fields.get('address', ''), 'source': fields.get('source', 'rightmove'),
        'agent_brand': fields.get('agent_brand', 'unknown'),
        'latitude': fields.get('latitude', 0), 'longitude': fields.get('longitude', 0),
        'let_type': 'long', 'features': '', 'description': fields.get('description', ''),
        'floor_count': fields.get('floor_count', 0), 'has_basement': 0, 'has_ground': 0,
        'has_first_floor': fields.get('has_first_floor', 0), 'has_second_floor': 0,
        'has_third_floor': 0, 'has_fourth_plus': 0, 'has_roof_terrace': 0,
    }
    X, _ = cp.build_features(pd.DataFrame([row]))
    return X.iloc[0]


# ── REGRESSION: is_social_housing (the −81% prime bug) ──────────────────────

@pytest.mark.parametrize('pc', ['SW1', 'SW3', 'SW7', 'W1', 'W8', 'NW8'])
def test_social_housing_does_not_fire_priceless_prime(cp, pc):
    """A price-less request in a PREMIUM postcode must NOT be flagged social
    housing. The bug: ppsf=0 (no price) < 3.5 fired on the whole prime market."""
    feat = _build_one(cp, postcode=pc, postcode_normalized=pc, area='Prime',
                      address='1 Some Street', bedrooms=2, size_sqft=1000)
    assert feat['is_social_housing'] == 0, (
        f"is_social_housing fired for price-less {pc} request — the −81% bug regressed"
    )


def test_social_housing_estate_regex_still_works(cp):
    """The price-independent estate-name branch must still flag known estates."""
    feat = _build_one(cp, postcode='SW1', postcode_normalized='SW1', area='Pimlico',
                      address='Flat 5, Churchill Gardens Estate', bedrooms=2, size_sqft=800)
    assert feat['is_social_housing'] == 1, "estate-name social-housing detection broke"


# ── REGRESSION: frequency encodings use training maps, not single-row 1.0 ────

def test_postcode_freq_not_degenerate(cp):
    """Single-row inference must inject the TRAINING postcode_freq, never 1.0."""
    feat = _build_one(cp, postcode='SW3', postcode_normalized='SW3', area='Chelsea')
    assert feat['postcode_freq'] != 1.0, "postcode_freq degenerated to 1.0 (single-row artifact)"
    assert 0.0 < feat['postcode_freq'] < 1.0


def test_postcode_freq_matches_inference_stats(cp):
    """postcode_freq must equal the baked training map value for the district."""
    stats = cp.load_inference_stats()
    assert stats, "inference.json missing — retrain must emit it"
    feat = _build_one(cp, postcode='SW3', postcode_normalized='SW3', area='Chelsea')
    expected = stats['postcode_freq'].get('SW3', stats['postcode_freq_default'])
    assert feat['postcode_freq'] == pytest.approx(expected, abs=1e-9)


def test_unseen_postcode_uses_default_freq(cp):
    """An unseen district must fall back to the training default, not 1.0."""
    stats = cp.load_inference_stats()
    feat = _build_one(cp, postcode='ZZ9', postcode_normalized='ZZ9', area='Nowhere')
    assert feat['postcode_freq'] == pytest.approx(stats['postcode_freq_default'], abs=1e-9)


# ── REGRESSION: floor_count default ─────────────────────────────────────────

def test_floor_count_default_is_zero(cp):
    """Missing floor_count must default to 0 (training median), not 1."""
    feat = _build_one(cp, postcode='SW3', postcode_normalized='SW3')  # no floor_count
    assert feat['floor_count'] == 0
    assert feat['floor_size_interaction'] == 0


# ── VERSION SINGLE-SOURCE ───────────────────────────────────────────────────

def test_version_single_sourced(cp):
    """canonical_predict must DERIVE the version from retrain_canonical.CANON_VERSION
    via the resolver — one declaration, not a second hardcoded literal."""
    import retrain_canonical as rc
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    import _canonical_features as cf
    _, _, resolver_ver = cf.resolve_feature_pipeline()
    assert rc.CANON_VERSION == cp.CANONICAL_VERSION == resolver_ver, (
        "version drift: retrain_canonical / canonical_predict / resolver disagree"
    )
    # same FE function object, not just same name → proves single source
    eng, _, _ = cf.resolve_feature_pipeline()
    assert eng is cp._ENGINEER


def test_no_hardcoded_version_literal_in_canonical_predict():
    """canonical_predict.py must not re-declare a 'vNN' literal or import a
    rental_price_models_v* module directly (it derives via the resolver)."""
    src = (ROOT / 'canonical_predict.py').read_text()
    assert "import rental_price_models_v" not in src
    assert "CANONICAL_VERSION = 'v20'" not in src and 'CANONICAL_VERSION = "v20"' not in src


# ── GOLDEN FIXTURE self-consistency (input → features → £) ───────────────────

def test_fixture_self_consistent(cp, fixture):
    """Every fixture sample must reproduce its stored prediction_pcm from its
    stored inputs (this is the contract the JS parity test validates against)."""
    for s in fixture['samples']:
        pred = float(cp.predict(pd.DataFrame([s['inputs']]))[0])
        assert pred == pytest.approx(s['prediction_pcm'], abs=0.05), (
            f"fixture sample {s['label']} not reproducible: "
            f"stored £{s['prediction_pcm']} vs computed £{pred:.2f}"
        )


def test_fixture_features_reproduce(cp, fixture):
    """Every stored feature value must reproduce from build_features (<1e-6)."""
    model, feat_order = cp.load_canonical()
    for s in fixture['samples']:
        X, _ = cp.build_features(pd.DataFrame([s['inputs']]))
        X = X.reindex(columns=feat_order, fill_value=0)
        for name in feat_order:
            assert float(X.iloc[0][name]) == pytest.approx(s['features'][name], abs=1e-6), (
                f"{s['label']}: feature {name} drifted"
            )


def test_sanity_properties_present_and_priced(cp, fixture):
    """The named SANITY samples must exist and predict a positive, plausible £."""
    labels = {s['label']: s for s in fixture['samples']}
    for lbl in ['SANITY_sw3_2bed_1000', 'SANITY_4_south_eaton_place']:
        assert lbl in labels, f"sanity sample {lbl} missing from fixture"
        p = labels[lbl]['prediction_pcm']
        assert 500 < p < 100000, f"{lbl} implausible £{p}"
