"""test_for_sale_inc3_model.py — TDD contract for the INC3 FULL SALE-PRICE MODEL.

WHAT THIS FILE PINS (the for-sale vertical, increment 3)
--------------------------------------------------------
Inc1 shipped the isolated for-sale data layer; Inc2 wired the CLI + Playwright sale
modes; Inc3 is the FULL sale-price model — the for-sale analogue of the rental v20
model, but a SEPARATE module family under for_sale/, trained on the COMMITTED
deterministic sale sample (no DB, no network, seed=42). This file is the RED-FIRST
contract for the new training harness (for_sale/sale_retrain.py), the lazy-load predict
convenience (for_sale/sale_predict.py), the freq-map / inference-mode plumbing in
for_sale/sale_price_model.py, and the tenure-agnostic feature primitives in
for_sale/sale_features.py.

It mirrors the rental retrain/serving PATTERNS by VALUE — it does NOT import the rental
chain (rental_price_models_v20 / canonical_predict / retrain_canonical / the rental
sidecar generators). The ISOLATION GUARDS at the bottom enforce that, plus the
sale-named artifact-path separation (never chrome-extension/api/).

CI-SAFETY: marker `for_sale` (registered, strict_markers). Trains on the committed
sale_training_sample.json (300 rows, 9 fields) — no network, no live DB, seed=42 — so it
ALWAYS runs and gates the PR. On the anti-silent-skip allowlist
(tests/test_ci_critical_tests_run.py). All artifact-writing tests use tmp_path +
monkeypatched output paths and NEVER pollute the repo's output/.

ZERO RENTAL REGRESSION: nothing here imports or mutates the rental model chain, the
parity-gated xgboost.js, the rental artifact, or property_scraper/items.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.for_sale

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "for_sale" / "sale_training_sample.json"

# Sale-magnitude bounds (re-stated locally to keep the test self-describing).
MIN_SALE_PRICE = 50_000
MAX_SALE_PRICE = 250_000_000


# ── Lazily-imported SUT modules (so collection is clean even before they exist; the
#    import error surfaces inside the test, as an UNIMPLEMENTED-PRODUCTION failure) ────

@pytest.fixture(scope="module")
def model_mod():
    from for_sale import sale_price_model
    return sale_price_model


@pytest.fixture(scope="module")
def retrain_mod():
    from for_sale import sale_retrain
    return sale_retrain


@pytest.fixture(scope="module")
def predict_mod():
    from for_sale import sale_predict
    return sale_predict


@pytest.fixture(scope="module")
def features_mod():
    from for_sale import sale_features
    return sale_features


@pytest.fixture(scope="module")
def sample_rows():
    rows = json.loads(SAMPLE.read_text())
    assert isinstance(rows, list) and rows
    return rows


@pytest.fixture(scope="module")
def trained(retrain_mod, sample_rows):
    """One deterministic training on the committed sample (seed=42). Module-scoped so the
    ~300-row train runs once for the whole file."""
    return retrain_mod.train(sample_rows, seed=42)


def _redirect_outputs(monkeypatch, retrain_mod, tmp_path):
    """Point every artifact path the harness writes at tmp_path so the repo's output/ is
    never polluted. Returns the tmp output dir. Mirrors the production constant names in
    for_sale/sale_retrain.py (spec section 2.1)."""
    out = tmp_path / "output"
    api = out / "sale_api"
    api.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(retrain_mod, "OUTPUT_DIR", out, raising=False)
    monkeypatch.setattr(retrain_mod, "MODEL_PATH", out / "sale_model.pkl", raising=False)
    monkeypatch.setattr(retrain_mod, "FEATURES_PATH", out / "sale_model_features.pkl", raising=False)
    monkeypatch.setattr(retrain_mod, "META_PATH", out / "sale_model_meta.json", raising=False)
    monkeypatch.setattr(retrain_mod, "INFERENCE_PATH", out / "sale_model_inference.json", raising=False)
    monkeypatch.setattr(retrain_mod, "SALE_API_DIR", api, raising=False)
    monkeypatch.setattr(retrain_mod, "GOLDEN_PATH", out / "sale_feature_parity_golden.json", raising=False)
    # Point SALES_DB at a guaranteed-ABSENT path so run_sale_retrain (called below with the
    # default db_path=SALES_DB) deterministically takes the synthetic branch REGARDLESS of any
    # leftover scratch output/sales.db on disk. The call sites pass allow_synthetic=True, the
    # explicit unit-test opt-in, so the synthetic refusal does not fire.
    monkeypatch.setattr(retrain_mod, "SALES_DB", tmp_path / "absent_sales.db", raising=False)
    return out


# ══════════════════════════════════════════════════════════════════════════════════════
# TRAIN — the additive return-dict (freq maps + seed), gate G6/G2 base.
# ══════════════════════════════════════════════════════════════════════════════════════

def test_train_returns_freq_maps_and_seed(trained):
    """train() now returns the baked freq sidecar inputs: a non-empty district freq_map,
    a NUMERIC freq_default == min(map), the postcode-area freq map + its default, and the
    seed (spec section 2.3 / gate G6)."""
    assert isinstance(trained.get("freq_map"), dict) and trained["freq_map"], \
        "freq_map must be a non-empty dict"
    assert trained["freq_default"] == min(trained["freq_map"].values()), \
        "freq_default must equal min(freq_map.values())"
    assert isinstance(trained.get("area_freq_map"), dict) and trained["area_freq_map"], \
        "area_freq_map must be a non-empty dict"
    assert trained["area_freq_default"] == min(trained["area_freq_map"].values()), \
        "area_freq_default must equal min(area_freq_map.values())"
    assert trained["seed"] == 42


# ══════════════════════════════════════════════════════════════════════════════════════
# RUN_SALE_RETRAIN — the full artifact bundle (gate G3/G6), written to tmp_path.
# ══════════════════════════════════════════════════════════════════════════════════════

def test_run_sale_retrain_writes_all_artifacts(monkeypatch, retrain_mod, tmp_path):
    """run_sale_retrain(write=True) produces the complete Inc4-unblocking bundle under the
    (monkeypatched-to-tmp) output dir: pickle pair, meta + inference sidecars, the
    sale_api Booster + features JSON, and the parity golden (gate G3/G6)."""
    out = _redirect_outputs(monkeypatch, retrain_mod, tmp_path)
    retrain_mod.run_sale_retrain(sample_path=SAMPLE, seed=42, write=True, allow_synthetic=True)
    for rel in (
        "sale_model.pkl",
        "sale_model_features.pkl",
        "sale_model_meta.json",
        "sale_model_inference.json",
        "sale_api/model.json",
        "sale_api/features.json",
        "sale_feature_parity_golden.json",
    ):
        assert (out / rel).exists(), f"run_sale_retrain did not write {rel}"


def test_inference_json_default_shape_invariant(monkeypatch, retrain_mod, tmp_path):
    """The baked sidecar enforces the _assert_default_shape invariant (mirrored by value):
    each top-level *_default is a NUMBER == min(map), and the map embeds NO nested
    'default' key — for BOTH district_freq and postcode_area_freq (gate G3)."""
    out = _redirect_outputs(monkeypatch, retrain_mod, tmp_path)
    retrain_mod.run_sale_retrain(sample_path=SAMPLE, seed=42, write=True, allow_synthetic=True)
    stats = json.loads((out / "sale_model_inference.json").read_text())
    for base in ("district_freq", "postcode_area_freq"):
        fmap, dflt = stats[base], stats[f"{base}_default"]
        assert isinstance(dflt, (int, float)) and not isinstance(dflt, bool), \
            f"{base}_default must be numeric"
        assert float(dflt) == float(min(fmap.values())), \
            f"{base}_default must equal min({base}.values())"
        assert "default" not in fmap, f"{base} map must not embed a nested 'default' key"


def test_meta_json_structural_backstop(monkeypatch, retrain_mod, trained, tmp_path):
    """The meta backstop (mirror retrain_canonical n_samples/db_source): n_features ==
    len(features) AND > 18 (FE genuinely grew); n_train == trained['n_train']; db_source
    is DERIVED from the load source (ENDS with the sample filename, NOT a hardcoded
    constant — the Retrain-readiness V2 footgun); canonical_version == 'sale_v1' (G6)."""
    out = _redirect_outputs(monkeypatch, retrain_mod, tmp_path)
    retrain_mod.run_sale_retrain(sample_path=SAMPLE, seed=42, write=True, allow_synthetic=True)
    meta = json.loads((out / "sale_model_meta.json").read_text())
    feats = json.loads((out / "sale_api" / "features.json").read_text())
    assert meta["n_features"] == len(feats)
    assert meta["n_features"] > 18, "meta n_features did not exceed the 18 baseline"
    assert meta["n_train"] == trained["n_train"]
    assert str(meta["db_source"]).endswith(SAMPLE.name), \
        f"db_source must be DERIVED from the load source, got {meta['db_source']!r}"
    assert meta["canonical_version"] == "sale_v1"


def test_sale_api_model_json_is_booster_and_features_match(monkeypatch, retrain_mod, tmp_path):
    """sale_api/model.json is a BOOSTER JSON (has a 'learner' key — the only format a JS
    tree-walker can read; NOT the sklearn-wrapper pickle), and sale_api/features.json is
    the feature_cols list in EXACT order (gate G3 / Inc4 unblock)."""
    out = _redirect_outputs(monkeypatch, retrain_mod, tmp_path)
    result = retrain_mod.run_sale_retrain(sample_path=SAMPLE, seed=42, write=True, allow_synthetic=True)
    booster = json.loads((out / "sale_api" / "model.json").read_text())
    assert "learner" in booster, "sale_api/model.json is not Booster JSON (no 'learner' key)"
    feats = json.loads((out / "sale_api" / "features.json").read_text())
    assert feats == list(result["feature_cols"]), \
        "features.json must equal feature_cols in the same order"


def test_artifact_round_trip_identical(monkeypatch, retrain_mod, model_mod, tmp_path):
    """Write the model to tmp via the harness, reload it, and predict on a fixed row
    BIT-IDENTICAL (np.isclose rtol=0, atol=1e-9) to the in-memory model — the serving
    artifact is faithful (gate G3)."""
    out = _redirect_outputs(monkeypatch, retrain_mod, tmp_path)
    result = retrain_mod.run_sale_retrain(sample_path=SAMPLE, seed=42, write=True, allow_synthetic=True)
    in_mem = result["model"]
    cols = result["feature_cols"]

    reloaded, cols2 = model_mod.load_model(out / "sale_model.pkl", out / "sale_model_features.pkl")
    assert cols2 == list(cols)

    row = dict(postcode="SW7", bedrooms=3, bathrooms=2, size_sqft=1400,
               property_type="Flat", address="Onslow Square, London, SW7")
    p_mem = model_mod.predict_one(in_mem, cols, **row)
    p_disk = model_mod.predict_one(reloaded, cols2, **row)
    assert np.isclose(p_mem, p_disk, rtol=0, atol=1e-9), (p_mem, p_disk)


# ══════════════════════════════════════════════════════════════════════════════════════
# PREDICT_ONE_DEFAULT — the lazy-load serving convenience (gate G2), Inc4 ergonomics.
# ══════════════════════════════════════════════════════════════════════════════════════

def test_predict_one_default_lazy_loads_and_returns_plausible(monkeypatch, retrain_mod, predict_mod, tmp_path):
    """predict_one_default lazy-loads the artifact and returns a dict whose
    predicted_price is a finite £ in the sane sale range; a normal SW3 3-bed flat with a
    real size is NOT low_confidence (gate G2)."""
    out = _redirect_outputs(monkeypatch, retrain_mod, tmp_path)
    retrain_mod.run_sale_retrain(sample_path=SAMPLE, seed=42, write=True, allow_synthetic=True)

    res = predict_mod.predict_one_default(
        postcode="SW3", bedrooms=3, bathrooms=2, size_sqft=1200, property_type="flat",
        model_path=out / "sale_model.pkl",
        features_path=out / "sale_model_features.pkl",
        inference_path=out / "sale_model_inference.json",
    )
    assert isinstance(res, dict)
    price = res["predicted_price"]
    assert np.isfinite(price) and MIN_SALE_PRICE <= price <= MAX_SALE_PRICE, price
    assert res["low_confidence"] is False
    assert res["estimated_size"] is False


def test_predict_one_default_missing_postcode_low_confidence(monkeypatch, retrain_mod, predict_mod, tmp_path):
    """postcode=None → district 'UNKNOWN', a finite plausible price, low_confidence True,
    no crash (gate G2 / Finding 3 #6)."""
    out = _redirect_outputs(monkeypatch, retrain_mod, tmp_path)
    retrain_mod.run_sale_retrain(sample_path=SAMPLE, seed=42, write=True, allow_synthetic=True)

    res = predict_mod.predict_one_default(
        postcode=None, bedrooms=2, bathrooms=2, size_sqft=900, property_type="flat",
        model_path=out / "sale_model.pkl",
        features_path=out / "sale_model_features.pkl",
        inference_path=out / "sale_model_inference.json",
    )
    assert res["district"] == "UNKNOWN"
    assert np.isfinite(res["predicted_price"])
    assert MIN_SALE_PRICE <= res["predicted_price"] <= MAX_SALE_PRICE
    assert res["low_confidence"] is True


def test_predict_one_default_missing_size_flags_estimated(monkeypatch, retrain_mod, predict_mod, tmp_path):
    """A missing/zero size_sqft → estimated_size True AND low_confidence True, with a
    still-finite plausible price (gate G2 / UX low-confidence guard)."""
    out = _redirect_outputs(monkeypatch, retrain_mod, tmp_path)
    retrain_mod.run_sale_retrain(sample_path=SAMPLE, seed=42, write=True, allow_synthetic=True)

    res = predict_mod.predict_one_default(
        postcode="SW3", bedrooms=2, bathrooms=2, size_sqft=0, property_type="flat",
        model_path=out / "sale_model.pkl",
        features_path=out / "sale_model_features.pkl",
        inference_path=out / "sale_model_inference.json",
    )
    assert res["estimated_size"] is True
    assert res["low_confidence"] is True
    assert np.isfinite(res["predicted_price"])
    assert MIN_SALE_PRICE <= res["predicted_price"] <= MAX_SALE_PRICE


# ══════════════════════════════════════════════════════════════════════════════════════
# INFERENCE-MODE FREQ — the BLOCKER-1 fix (baked map, not per-frame 1.0), non-vacuous.
# ══════════════════════════════════════════════════════════════════════════════════════

def test_inference_mode_district_freq_not_degenerate(model_mod):
    """build_features(inference=True, freq_map=baked) uses the BAKED training-distribution
    frequency for a single row, NOT the degenerate per-frame 1.0 that a single-row frame
    yields. Assert the two paths DIFFER — proving the BLOCKER-1 fix is live and
    non-vacuous (Finding 5)."""
    row = {"postcode": "SW3", "bedrooms": 2, "bathrooms": 2, "size_sqft": 900,
           "property_type": "Flat", "address": "Cadogan Gardens, London, SW3",
           "asking_price": 3_000_000}
    baked = {"SW3": 0.07, "SE15": 0.03}
    default = min(baked.values())

    X_inf, _ = model_mod.build_features(
        [row], inference=True, freq_map=baked, freq_default=default
    )
    X_frame, _ = model_mod.build_features([row])  # per-frame: single row → 1.0

    assert X_inf["district_freq"].iloc[0] == baked["SW3"], \
        "inference mode must use the baked freq for the district"
    assert X_frame["district_freq"].iloc[0] == 1.0, \
        "per-frame single-row district_freq is the degenerate 1.0 (the bug being fixed)"
    assert X_inf["district_freq"].iloc[0] != X_frame["district_freq"].iloc[0], \
        "BLOCKER-1 fix is vacuous — baked and per-frame freq are identical"


def test_n_features_in_assert_catches_drift(monkeypatch, retrain_mod, predict_mod, model_mod, tmp_path):
    """The lazy loader asserts model.n_features_in_ == len(feature_cols) (mirror
    canonical_predict :189): feeding a feature_cols list with one EXTRA name must raise
    ValueError on load, never silently mispredict."""
    out = _redirect_outputs(monkeypatch, retrain_mod, tmp_path)
    result = retrain_mod.run_sale_retrain(sample_path=SAMPLE, seed=42, write=True, allow_synthetic=True)

    # Save a deliberately DRIFTED features file (one extra column name) next to the model.
    bad_features = out / "drift_features.pkl"
    import pickle
    with open(bad_features, "wb") as f:
        pickle.dump(list(result["feature_cols"]) + ["__bogus_extra_col__"], f)

    # Defeat any module-level load cache so the drift is actually re-checked.
    if hasattr(predict_mod, "_CACHE"):
        try:
            predict_mod._CACHE.clear()
        except Exception:
            monkeypatch.setattr(predict_mod, "_CACHE", {}, raising=False)

    with pytest.raises(ValueError):
        predict_mod.predict_one_default(
            postcode="SW3", bedrooms=2, bathrooms=1, size_sqft=800, property_type="flat",
            model_path=out / "sale_model.pkl",
            features_path=bad_features,
            inference_path=out / "sale_model_inference.json",
        )


# ══════════════════════════════════════════════════════════════════════════════════════
# DETERMINISM / STABILITY / STRUCTURAL ORDERING GATES (G5 / G7 / G4 / R²).
# ══════════════════════════════════════════════════════════════════════════════════════

def test_determinism_two_trainings_identical(retrain_mod, sample_rows):
    """Two in-process trainings at seed=42 produce BYTE-IDENTICAL metrics AND an identical
    prediction on a fixed row (gate G5; n_jobs=1 + fixed seed). The monotone constraint +
    new features must NOT break the determinism the baseline already had."""
    from for_sale import sale_price_model as spm
    a = retrain_mod.train(sample_rows, seed=42)
    b = retrain_mod.train(sample_rows, seed=42)
    assert a["metrics"] == b["metrics"], (a["metrics"], b["metrics"])

    row = dict(postcode="SW7", bedrooms=3, bathrooms=2, size_sqft=1400,
               property_type="Flat", address="Onslow Square, London, SW7")
    pa = spm.predict_one(a["model"], a["feature_cols"], **row)
    pb = spm.predict_one(b["model"], b["feature_cols"], **row)
    assert pa == pb, (pa, pb)


def test_seed_stability_low_variance(retrain_mod, sample_rows):
    """seed_stability over (42,7,123) shows the model is not seed-lucky: r2_std < 0.05 and
    r2_min > 0.6 (gate G7)."""
    res = retrain_mod.seed_stability(sample_rows, seeds=(42, 7, 123))
    assert res["r2_std"] < 0.05, f"R² varies too much across seeds: std={res['r2_std']}"
    assert res["r2_min"] > 0.6, f"a seed under-fit: r2_min={res['r2_min']}"


def test_monotone_prime_ge_ordinary(model_mod, trained):
    """A prime district (SW1X) predicts >= an ordinary district (SE15), all else equal —
    a LEARNED ordering on this fixture (gate G4b; not a hard monotone constraint)."""
    base = dict(bedrooms=3, bathrooms=2, size_sqft=1200, property_type="Flat",
                address="A Street, London")
    prime = model_mod.predict_one(trained["model"], trained["feature_cols"],
                                  postcode="SW1X", **base)
    ordinary = model_mod.predict_one(trained["model"], trained["feature_cols"],
                                     postcode="SE15", **base)
    assert prime >= ordinary, f"prime {prime} < ordinary {ordinary}"


def test_new_build_changes_price(model_mod, trained):
    """is_new_build=1 vs 0, all else equal, moves the prediction by a non-zero delta —
    the sale-only feature is wired into the model (gate G4c)."""
    base = dict(postcode="SW3", bedrooms=2, bathrooms=2, size_sqft=1000,
                property_type="Flat", address="A Street, London, SW3")
    new = model_mod.predict_one(trained["model"], trained["feature_cols"],
                                is_new_build=1, **base)
    old = model_mod.predict_one(trained["model"], trained["feature_cols"],
                                is_new_build=0, **base)
    assert abs(new - old) > 0, "is_new_build had zero effect — feature not wired in"


def test_baseline_r2_smoke(trained):
    """The pipeline recovers the synthetic formula's signal: held-out R² > 0.6 (a
    signal-recovery contract, NOT a real-world accuracy claim)."""
    assert trained["metrics"]["r2"] > 0.6, trained["metrics"]["r2"]


# ══════════════════════════════════════════════════════════════════════════════════════
# ISOLATION GUARDS — extend the rental ban to the new Inc3 modules + the sale artifacts.
# ══════════════════════════════════════════════════════════════════════════════════════

# Banned rental-chain import statements (Inc3 ADDS retrain_canonical to the baseline ban).
_BANNED_RENTAL_IMPORTS = (
    "import rental_price_models_v20",
    "from rental_price_models_v20",
    "import canonical_predict",
    "from canonical_predict",
    "import retrain_canonical",
    "from retrain_canonical",
)

_FOR_SALE_SRC_FILES = (
    ROOT / "for_sale" / "sale_price_model.py",
    ROOT / "for_sale" / "sale_features.py",
    ROOT / "for_sale" / "sale_retrain.py",
    ROOT / "for_sale" / "sale_predict.py",
)


def test_inc3_modules_do_not_import_rental_chain():
    """Each of the 4 for-sale source files must contain NONE of the banned rental-chain
    import statements (extends the baseline grep guard, ADDING retrain_canonical)."""
    for path in _FOR_SALE_SRC_FILES:
        assert path.exists(), f"expected for-sale source file missing: {path}"
        src = path.read_text()
        for banned in _BANNED_RENTAL_IMPORTS:
            assert banned not in src, f"{path.name} illegally couples to the rental chain: {banned}"


def test_inc3_artifact_paths_separate_from_rental(retrain_mod):
    """Every Inc3 artifact path contains 'sale' and NONE contains 'rental_model'; the
    sale API dir is under output/sale_api, NEVER chrome-extension/api (the rental dir)."""
    paths = {
        "MODEL_PATH": retrain_mod.MODEL_PATH,
        "META_PATH": retrain_mod.META_PATH,
        "INFERENCE_PATH": retrain_mod.INFERENCE_PATH,
        "SALE_API_DIR": retrain_mod.SALE_API_DIR,
        "GOLDEN_PATH": retrain_mod.GOLDEN_PATH,
    }
    for name, p in paths.items():
        s = str(p)
        assert "sale" in s, f"{name} path is not sale-named: {s}"
        assert "rental_model" not in s, f"{name} points at the rental artifact: {s}"
    api = str(retrain_mod.SALE_API_DIR).replace("\\", "/")
    assert api.endswith("output/sale_api") or "/output/sale_api" in api, \
        f"SALE_API_DIR must be output/sale_api, got {api}"
    assert "chrome-extension/api" not in api, "SALE_API_DIR must NOT be the rental chrome-extension/api"


def test_inc3_only_legal_shared_import():
    """The ONLY non-stdlib / non-(numpy|pandas|sklearn|xgboost) cross-package import any
    for-sale source may make is property_scraper.services.fingerprint — and only
    sale_data.py uses it. No for-sale module reaches into the rental package."""
    import re as _re

    allowed_third_party_prefixes = ("numpy", "pandas", "sklearn", "xgboost", "scipy")
    import_re = _re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", _re.MULTILINE)

    # sale_data.py legitimately holds the one legal shared import; the 4 model-side files
    # must NOT reach into property_scraper at all.
    for path in _FOR_SALE_SRC_FILES:
        assert path.exists(), f"expected for-sale source file missing: {path}"
        src = path.read_text()
        for mod in import_re.findall(src):
            top = mod.split(".")[0]
            if top in ("for_sale", "__future__"):
                continue
            if top in allowed_third_party_prefixes:
                continue
            # stdlib / local helpers are fine; the hard ban is on the rental MODEL package
            # and on any property_scraper reach-in from the model-side files.
            assert top != "property_scraper", (
                f"{path.name} must not import property_scraper (only sale_data.py may, "
                f"for the fingerprint primitive): {mod}"
            )
            for banned in ("rental_price_models_v20", "canonical_predict", "retrain_canonical"):
                assert banned not in mod, f"{path.name} couples to rental chain via import {mod}"
