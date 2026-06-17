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


def test_unseen_postcode_area_uses_default_freq(cp):
    """The AREA frequency must ALSO fall back to its training default for an unseen
    area — not just the district freq. (Problem 3: the default must apply for BOTH
    postcode_freq AND postcode_area_freq.)"""
    stats = cp.load_inference_stats()
    feat = _build_one(cp, postcode='ZZ9', postcode_normalized='ZZ9', area='Nowhere')
    assert feat['postcode_area_freq'] == pytest.approx(
        stats['postcode_area_freq_default'], abs=1e-9
    )


def test_unseen_postcode_default_is_numeric_no_none_reaches_model(cp):
    """The bug (CI np.log1p TypeError) was a None default landing in an object-dtype
    column → np.log1p blew up. Lock in that an unseen-postcode feature row is fully
    NUMERIC and finite: no None/object/NaN can reach the model for ANY feature."""
    import numpy as np
    model, feat_order = cp.load_canonical()
    X, _ = cp.build_features(pd.DataFrame([{
        'bedrooms': 2, 'bathrooms': 1, 'size_sqft': 900,
        'postcode': 'ZZ9 9ZZ', 'postcode_normalized': 'ZZ9', 'area': 'Nowhere',
        'property_type': 'flat', 'property_type_std': 'flat',
        'address': '1 Unseen Road', 'source': 'rightmove', 'agent_brand': 'unknown',
        'latitude': 51.49, 'longitude': -0.168, 'let_type': 'long',
        'features': '', 'description': '', 'floor_count': 0,
        'has_basement': 0, 'has_ground': 0, 'has_first_floor': 0,
        'has_second_floor': 0, 'has_third_floor': 0, 'has_fourth_plus': 0,
        'has_roof_terrace': 0,
    }]))
    Xr = X.reindex(columns=feat_order, fill_value=0)
    # No object dtype anywhere (object-dtype is what made np.log1p raise).
    assert not any(str(dt) == 'object' for dt in Xr.dtypes), (
        f"object-dtype column(s) survived to inference: "
        f"{[c for c, dt in Xr.dtypes.items() if str(dt) == 'object']}"
    )
    # Every value finite (no None→NaN, no inf) — and the freq defaults specifically
    # are the real numeric training defaults, not None and not 1.0.
    vals = Xr.iloc[0].to_numpy(dtype=float)
    assert np.isfinite(vals).all(), "non-finite value reached the model for an unseen postcode"
    stats = cp.load_inference_stats()
    assert Xr.iloc[0]['postcode_freq'] == pytest.approx(stats['postcode_freq_default'], abs=1e-9)
    assert Xr.iloc[0]['postcode_freq'] != 1.0
    # The model must actually predict a finite, positive £ for the unseen row.
    pred = float(np.expm1(model.predict(Xr))[0])
    assert math.isfinite(pred) and pred > 0, f"unseen-postcode prediction not sane: £{pred}"


# ── REGRESSION: coordless-row distance dtype (the REAL np.log1p CI crash) ────

def test_coordless_row_distance_is_numeric_no_log1p_crash(cp):
    """The ACTUAL CI failure (run 27636202638): a single COORDLESS row (no lat/long,
    unseen district → no centroid) made v20's distance block produce an all-None
    object column → median()=NaN → fillna no-op → np.log1p(object) TypeError.

    This reproduces it deterministically (conftest sets future.no_silent_downcasting
    so local == CI). build_features must NOT raise, the distance features must be
    NUMERIC and finite, and (parity with JS, which uses CITY_CENTER for missing coords)
    center_distance_km must be 0 for a coordless row."""
    import numpy as np
    # latitude=longitude=0 is how the inference path (predict_one) represents "no
    # coords"; ZZ9 → no POSTCODE_CENTROIDS hit → coordless after centroid fill.
    X, _ = cp.build_features(pd.DataFrame([{
        'bedrooms': 2, 'bathrooms': 1, 'size_sqft': 900,
        'postcode': 'ZZ9 9ZZ', 'postcode_normalized': 'ZZ9', 'area': 'Nowhere',
        'property_type': 'flat', 'property_type_std': 'flat',
        'address': '1 Unseen Road', 'source': 'rightmove', 'agent_brand': 'unknown',
        'latitude': 0, 'longitude': 0, 'let_type': 'long',
        'features': '', 'description': '', 'floor_count': 0,
        'has_basement': 0, 'has_ground': 0, 'has_first_floor': 0,
        'has_second_floor': 0, 'has_third_floor': 0, 'has_fourth_plus': 0,
        'has_roof_terrace': 0,
    }]))
    row = X.iloc[0]
    for col in ('center_distance_km', 'tube_distance_km', 'log_center_distance',
                'log_tube_distance', 'center_distance_inv'):
        assert col in X.columns, f"{col} missing from features"
        assert str(X[col].dtype) != 'object', f"{col} is object-dtype (np.log1p will crash)"
        assert np.isfinite(float(row[col])), f"{col} not finite for a coordless row"
    # A coord-less row must resolve to the NEUTRAL training-median default distance
    # (DEFAULT_CENTER_DISTANCE_KM), NOT 0 km / city-centre. The JS predictor mirrors
    # the same constant — see test_js_python_default_distance_constants.
    import rental_price_models_v20 as v20
    assert float(row['center_distance_km']) == pytest.approx(v20.DEFAULT_CENTER_DISTANCE_KM, abs=1e-6), (
        "coordless row must use the neutral median default distance, not 0 (city centre)"
    )
    assert float(row['tube_distance_km']) == pytest.approx(v20.DEFAULT_TUBE_DISTANCE_KM, abs=1e-6)


# ── CONTRACT: gen_inference_stats writer output shape ───────────────────────

def test_inference_stats_default_keys_are_sibling_numeric(cp):
    """The writer (gen_inference_stats.py) must persist the freq DEFAULTS as numeric
    SIBLING keys (postcode_freq_default / postcode_area_freq_default), NOT as a nested
    `default` entry inside the freq map. A nested None `default` was the regression
    that crashed np.log1p; the reader keys off the sibling. Equal to min(map values)."""
    stats = cp.load_inference_stats()
    assert stats, "inference.json missing — retrain must emit it"
    for base in ('postcode_freq', 'postcode_area_freq'):
        fmap = stats[base]
        dflt = stats[f'{base}_default']
        # sibling default present and a real float (never None / str)
        assert isinstance(dflt, float), f"{base}_default must be float, got {type(dflt).__name__}"
        assert math.isfinite(dflt), f"{base}_default must be finite"
        # NO nested 'default' key polluting the map (the buggy shape)
        assert 'default' not in fmap, f"{base} map must not embed a nested 'default' key"
        # default == smallest observed training frequency (the documented contract)
        assert dflt == pytest.approx(min(fmap.values()), abs=1e-12), (
            f"{base}_default ({dflt}) must equal min observed freq ({min(fmap.values())})"
        )


def _parse_js_const_map(src: str, name: str) -> dict:
    """Extract a single-line `NAME: { ... }` JS object literal from xgboost.js as a
    Python dict (single→double quotes → JSON). Used to assert the JS freq DEFAULT
    matches Python's without executing JS."""
    import re
    m = re.search(rf'{name}:\s*(\{{[^}}]*\}})', src)
    assert m, f"could not locate {name} object literal in xgboost.js"
    return json.loads(m.group(1).replace("'", '"'))


def test_js_python_default_freq_invariant(cp):
    """LOAD-BEARING cross-language invariant (architect P2): the JS freq DEFAULT is
    stored IN-MAP under 'default' (chrome-extension/xgboost.js), Python keeps it as a
    top-level sibling. They are DELIBERATELY different SHAPES; parity holds only while
    the NUMBER matches:
        xgboost.js POSTCODE_FREQ['default'] == inference.json postcode_freq_default
                                            == min(postcode_freq.values())
    A retrain that regenerates inference.json but not xgboost.js silently breaks this.
    This is the regression lock that was missing."""
    stats = cp.load_inference_stats()
    js_src = (ROOT / 'chrome-extension' / 'xgboost.js').read_text()
    js_pc = _parse_js_const_map(js_src, 'POSTCODE_FREQ')
    js_area = _parse_js_const_map(js_src, 'POSTCODE_AREA_FREQ')

    assert 'default' in js_pc, "xgboost.js POSTCODE_FREQ must carry an in-map 'default' key"
    assert 'default' in js_area, "xgboost.js POSTCODE_AREA_FREQ must carry an in-map 'default' key"

    # district default: JS in-map == Python sibling == min(training freqs)
    assert js_pc['default'] == pytest.approx(stats['postcode_freq_default'], abs=1e-12), (
        f"JS POSTCODE_FREQ['default']={js_pc['default']} != Python "
        f"postcode_freq_default={stats['postcode_freq_default']} — regen xgboost.js from inference.json"
    )
    assert js_pc['default'] == pytest.approx(min(stats['postcode_freq'].values()), abs=1e-12)
    # area default: same three-way equality
    assert js_area['default'] == pytest.approx(stats['postcode_area_freq_default'], abs=1e-12), (
        f"JS POSTCODE_AREA_FREQ['default']={js_area['default']} != Python "
        f"postcode_area_freq_default={stats['postcode_area_freq_default']}"
    )
    assert js_area['default'] == pytest.approx(min(stats['postcode_area_freq'].values()), abs=1e-12)


def test_js_python_default_distance_constants(cp):
    """Coord-less rows use neutral DEFAULT distance constants (training medians). The JS
    predictor MUST carry the SAME numeric constants as Python
    rental_price_models_v20.DEFAULT_{CENTER,TUBE}_DISTANCE_KM, or coord-less requests
    (very common — most /api/predict + extension calls have no lat/long) silently
    diverge. The golden fixture's coordless sample also gates this, but this asserts the
    constants directly so a JS edit can't drift them."""
    import re
    import rental_price_models_v20 as v20
    js_src = (ROOT / 'chrome-extension' / 'xgboost.js').read_text()

    def _js_num(key):
        m = re.search(rf'{key}:\s*([0-9.]+)', js_src)
        assert m, f"{key} not found in xgboost.js"
        return float(m.group(1))

    assert _js_num('DEFAULT_CENTER_DISTANCE_KM') == pytest.approx(
        v20.DEFAULT_CENTER_DISTANCE_KM, abs=1e-12), "JS center-distance default drifted from Python"
    assert _js_num('DEFAULT_TUBE_DISTANCE_KM') == pytest.approx(
        v20.DEFAULT_TUBE_DISTANCE_KM, abs=1e-12), "JS tube-distance default drifted from Python"


def test_both_python_writers_emit_identical_default_shape():
    """The TWO Python writers (gen_inference_stats.py AND retrain_canonical.py ~186-199)
    must stay shape-identical: both emit sibling float `*_default` keys and NEITHER
    embeds a nested 'default' in the freq map. Asserted statically on source so the
    test needs no DB/retrain."""
    gis = (ROOT / 'gen_inference_stats.py').read_text()
    rc = (ROOT / 'retrain_canonical.py').read_text()
    for src, who in ((gis, 'gen_inference_stats.py'), (rc, 'retrain_canonical.py')):
        assert "'postcode_freq_default': float(min(pc_freq.values()))" in src, (
            f"{who} must emit sibling postcode_freq_default = float(min(pc_freq.values()))"
        )
        assert "'postcode_area_freq_default': float(min(area_freq.values()))" in src, (
            f"{who} must emit sibling postcode_area_freq_default = float(min(area_freq.values()))"
        )
        # Neither writer may inject a nested 'default' into the freq maps.
        assert "pc_freq['default']" not in src and 'pc_freq["default"]' not in src, (
            f"{who} must not embed a nested 'default' in postcode_freq"
        )


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
