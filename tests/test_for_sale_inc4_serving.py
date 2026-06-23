"""test_for_sale_inc4_serving.py — TDD contract for INC4a FOR-SALE SERVING (Python side).

WHAT THIS FILE PINS (the for-sale vertical, increment 4a, Group A — js-predictor)
---------------------------------------------------------------------------------
Inc3 shipped the full sale model + a regenerable artifact contract. Inc4a adds the JS
serving predictor + the 0/0 byte-parity gate, the routes, and the extension mode. This
file is the PYTHON half of the Inc4a RED test list (spec section 7):

  * the artifact-regen path is live and network-free (4 JSONs, 34 features);
  * gen_sale_golden.py SCORES the COMMITTED Booster (NOT a retrain) — sample0's
    prediction_price == np.expm1(Booster.predict) within 1e-9;
  * the committed golden uses the inference=True BAKED-MAP path (BLOCKER 1 fix) — the
    'unseen_district_default_freq' sample's district_freq equals the baked
    district_freq_default (e.g. 0.097), NOT the degenerate 1.0 (AMENDMENT FIX 2);
  * the golden's £ key is `prediction_price`, NOT `prediction_pcm` (guards the gate fork);
  * the inference sidecar's *_default shape is the numeric min(map), no nested 'default';
  * Inc4 artifacts live under output/sale_api/ — NEVER chrome-extension/api/ (rental dir).

It mirrors the rental serving PATTERNS by VALUE and imports NONE of the rental chain.
CI-SAFETY: marker `for_sale` (registered, strict_markers); deterministic, network-free.
ZERO RENTAL REGRESSION: nothing here imports or mutates the rental model chain, the
parity-gated xgboost.js, the rental artifacts, or property_scraper/items.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.for_sale

ROOT = Path(__file__).resolve().parent.parent
SALE_API_DIR = ROOT / "output" / "sale_api"
MODEL_JSON = SALE_API_DIR / "model.json"
FEATURES_JSON = SALE_API_DIR / "features.json"
INFERENCE_JSON = ROOT / "output" / "sale_model_inference.json"
GOLDEN_JSON = ROOT / "output" / "sale_feature_parity_golden.json"
GEN_SALE_GOLDEN = ROOT / "gen_sale_golden.py"


# ──────────────────────────────────────────────────────────────────────────────────────
# 1. Artifact regen is live + network-free (keeps Inc4 honest about the regen path).
# ──────────────────────────────────────────────────────────────────────────────────────

def test_inc4_artifacts_regenerate_network_free(tmp_path, monkeypatch):
    """run_sale_retrain(write=True) writes all 4 JSONs under a tmp OUTPUT_DIR redirect;
    model.json carries learner.gradient_booster.model.trees and features.json has len==34."""
    from for_sale import sale_retrain as rt

    out = tmp_path / "output"
    api = out / "sale_api"
    api.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rt, "OUTPUT_DIR", out, raising=False)
    monkeypatch.setattr(rt, "MODEL_PATH", out / "sale_model.pkl", raising=False)
    monkeypatch.setattr(rt, "FEATURES_PATH", out / "sale_model_features.pkl", raising=False)
    monkeypatch.setattr(rt, "META_PATH", out / "sale_model_meta.json", raising=False)
    monkeypatch.setattr(rt, "INFERENCE_PATH", out / "sale_model_inference.json", raising=False)
    monkeypatch.setattr(rt, "SALE_API_DIR", api, raising=False)
    monkeypatch.setattr(rt, "GOLDEN_PATH", out / "sale_feature_parity_golden.json", raising=False)
    # Point SALES_DB at a guaranteed-ABSENT path so run_sale_retrain (default db_path=SALES_DB)
    # deterministically takes the synthetic branch REGARDLESS of any scratch output/sales.db;
    # allow_synthetic=True is the explicit unit-test opt-in past the production synthetic refusal.
    monkeypatch.setattr(rt, "SALES_DB", tmp_path / "absent_sales.db", raising=False)

    rt.run_sale_retrain(write=True, allow_synthetic=True)

    model = json.loads((api / "model.json").read_text())
    assert model["learner"]["gradient_booster"]["model"]["trees"], "Booster JSON must carry trees"
    features = json.loads((api / "features.json").read_text())
    assert len(features) == 34, f"features.json must have 34 names, got {len(features)}"
    assert (out / "sale_model_inference.json").exists()
    assert (out / "sale_feature_parity_golden.json").exists()


# ──────────────────────────────────────────────────────────────────────────────────────
# 2. gen_sale_golden.py SCORES the committed Booster (not a retrain).
# ──────────────────────────────────────────────────────────────────────────────────────

def test_inc4_gen_sale_golden_scores_committed_model():
    """gen_sale_golden.build_golden() loads output/sale_api/model.json as a Booster (NOT a
    retrain) and produces a golden whose sample0 prediction_price is finite > 0 and ==
    np.expm1(Booster.predict) within 1e-9."""
    import xgboost as xgb

    import gen_sale_golden as g

    golden = g.build_golden()
    s0 = golden["samples"][0]
    assert np.isfinite(s0["prediction_price"]) and s0["prediction_price"] > 0

    # Re-derive sample0 from the COMMITTED Booster + baked maps and confirm equality.
    from for_sale import sale_price_model
    from for_sale.sale_retrain import _GOLDEN_INPUTS

    inf = json.loads(INFERENCE_JSON.read_text())
    feature_cols = json.loads(FEATURES_JSON.read_text())
    booster = xgb.Booster()
    booster.load_model(str(MODEL_JSON))

    row = {k: v for k, v in _GOLDEN_INPUTS[0].items() if k != "label"}
    X, _ = sale_price_model.build_features(
        [row],
        inference=True,
        freq_map=inf["district_freq"],
        freq_default=inf["district_freq_default"],
        area_freq_map=inf["postcode_area_freq"],
        area_freq_default=inf["postcode_area_freq_default"],
    )
    X = X.reindex(columns=feature_cols, fill_value=0.0).astype(float)
    dmat = xgb.DMatrix(X.to_numpy(dtype=float), feature_names=list(feature_cols))
    expected = float(np.expm1(booster.predict(dmat))[0])
    assert abs(s0["prediction_price"] - expected) < 1e-9


def test_inc4_gen_sale_golden_runs_as_script_and_writes_committed_golden():
    """python3 gen_sale_golden.py runs network-free and (re)writes the committed golden with
    34 features / 'prediction_price' keyed samples."""
    proc = subprocess.run(
        [sys.executable, str(GEN_SALE_GOLDEN)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"gen_sale_golden.py failed: {proc.stderr}"
    golden = json.loads(GOLDEN_JSON.read_text())
    assert golden["n_features"] == 34
    assert all("prediction_price" in s for s in golden["samples"])


# ──────────────────────────────────────────────────────────────────────────────────────
# 3. Golden £ key is prediction_price (guards the gate fork delta).
# ──────────────────────────────────────────────────────────────────────────────────────

def test_inc4_golden_key_is_prediction_price():
    """Every committed-golden sample has key 'prediction_price' and NOT 'prediction_pcm'."""
    golden = json.loads(GOLDEN_JSON.read_text())
    assert golden["samples"], "golden must carry samples"
    for s in golden["samples"]:
        assert "prediction_price" in s, f"{s.get('label')} missing prediction_price"
        assert "prediction_pcm" not in s, f"{s.get('label')} must NOT carry prediction_pcm"


# ──────────────────────────────────────────────────────────────────────────────────────
# 4. AMENDMENT FIX 2 — committed golden uses inference=True baked-map path (NOT 1.0).
# ──────────────────────────────────────────────────────────────────────────────────────

def test_inc4_committed_golden_is_inference_true_not_degenerate():
    """The 'unseen_district_default_freq' sample's district_freq == the baked
    district_freq_default (e.g. ~0.097), NOT 1.0. A stale inference=False golden (which
    collapses district_freq to 1.0 on a single row) fails this loudly (AMENDMENT FIX 2)."""
    inf = json.loads(INFERENCE_JSON.read_text())
    default_d = float(inf["district_freq_default"])
    golden = json.loads(GOLDEN_JSON.read_text())

    by_label = {s["label"]: s for s in golden["samples"]}
    assert "unseen_district_default_freq" in by_label, "golden must keep the unseen-district sample"
    df = float(by_label["unseen_district_default_freq"]["features"]["district_freq"])
    assert df == pytest.approx(default_d), (
        f"unseen district_freq must be the baked default {default_d}, got {df} "
        f"(a 1.0 here means a stale inference=False golden was committed)"
    )
    assert df != 1.0, "district_freq must NOT be the degenerate single-row 1.0"

    # And at least one in-map district sample is non-default (proves the gate is non-vacuous).
    in_map = [
        s for s in golden["samples"]
        if float(s["features"]["district_freq"]) not in (1.0, default_d)
    ]
    assert in_map, "at least one golden sample must hit a baked in-map district_freq != default"


def test_inc4_committed_golden_no_sample_has_degenerate_freq():
    """No committed-golden sample carries the degenerate single-row district_freq /
    postcode_area_freq == 1.0 (the inference=False collapse)."""
    golden = json.loads(GOLDEN_JSON.read_text())
    for s in golden["samples"]:
        assert s["features"]["district_freq"] != 1.0, f"{s['label']} has degenerate district_freq=1.0"
        assert s["features"]["postcode_area_freq"] != 1.0, (
            f"{s['label']} has degenerate postcode_area_freq=1.0"
        )


# ──────────────────────────────────────────────────────────────────────────────────────
# 5. Coverage rows (BLOCKER 2 / FIX 3) actually exercise the branches.
# ──────────────────────────────────────────────────────────────────────────────────────

def test_inc4_golden_coverage_rows_exercise_branches():
    """The 4 added coverage rows exercise: a real haversine distance (!= coordless default),
    a POA qualifier (price_qualifier_poa==1), and a literal size_sqft==0 kept (NOT 700)."""
    from for_sale.sale_features import DEFAULT_CENTER_DISTANCE_KM

    golden = json.loads(GOLDEN_JSON.read_text())
    by_label = {s["label"]: s for s in golden["samples"]}

    # (b) real coords -> active haversine, distinct from the coordless DEFAULT.
    assert "real_coords_haversine" in by_label
    cdk = float(by_label["real_coords_haversine"]["features"]["center_distance_km"])
    assert abs(cdk - DEFAULT_CENTER_DISTANCE_KM) > 1e-6, "real-coords sample must NOT be the coordless default"

    # (c) POA qualifier -> price_qualifier_poa == 1.
    assert "poa_qualifier" in by_label
    assert by_label["poa_qualifier"]["features"]["price_qualifier_poa"] == 1.0

    # (d) literal size 0 -> kept as 0 (NOT mapped to 700).
    assert "size_zero_literal" in by_label
    assert by_label["size_zero_literal"]["features"]["size_sqft"] == 0.0

    # (a) in-map district -> a real baked freq (covered by the inference-true test); confirm key.
    assert "in_map_district_sw3" in by_label


def test_inc4_size_zero_literal_not_700():
    """FIX 4: build_features keeps a literal size_sqft=0 (only NaN/absent maps to 700)."""
    from for_sale import sale_price_model

    inf = json.loads(INFERENCE_JSON.read_text())
    X0, _ = sale_price_model.build_features(
        [{"postcode": "SW10", "bedrooms": 2, "bathrooms": 1, "size_sqft": 0, "property_type": "Flat"}],
        inference=True, freq_map=inf["district_freq"], freq_default=inf["district_freq_default"],
        area_freq_map=inf["postcode_area_freq"], area_freq_default=inf["postcode_area_freq_default"],
    )
    assert float(X0["size_sqft"].iloc[0]) == 0.0, "literal size_sqft=0 must stay 0"

    Xn, _ = sale_price_model.build_features(
        [{"postcode": "SW10", "bedrooms": 2, "bathrooms": 1, "property_type": "Flat"}],
        inference=True, freq_map=inf["district_freq"], freq_default=inf["district_freq_default"],
        area_freq_map=inf["postcode_area_freq"], area_freq_default=inf["postcode_area_freq_default"],
    )
    assert float(Xn["size_sqft"].iloc[0]) == 700.0, "absent size_sqft must map to the 700 fallback"


# ──────────────────────────────────────────────────────────────────────────────────────
# 6. Inference sidecar default shape (numeric min(map), no nested 'default').
# ──────────────────────────────────────────────────────────────────────────────────────

def test_inc4_inference_default_shape():
    """sale_model_inference.json: district_freq_default == min(district_freq.values()),
    postcode_area_freq_default == min(...), and neither map embeds a nested 'default' key."""
    inf = json.loads(INFERENCE_JSON.read_text())
    for base in ("district_freq", "postcode_area_freq"):
        fmap = inf[base]
        dflt = inf[f"{base}_default"]
        assert isinstance(dflt, float)
        assert dflt == float(min(fmap.values())), f"{base}_default must equal min({base}.values())"
        assert "default" not in fmap, f"{base} must not embed a nested 'default' key"


# ──────────────────────────────────────────────────────────────────────────────────────
# 7. Artifact-path isolation — sale lives in output/sale_api/, never chrome-extension/api/.
# ──────────────────────────────────────────────────────────────────────────────────────

def test_inc4_predict_sale_artifact_paths_isolated():
    """No Inc4 sale serving artifact path resolves under chrome-extension/api/ (the rental
    dir). The sale Booster/features live under output/sale_api/."""
    rental_api = (ROOT / "chrome-extension" / "api").resolve()
    for p in (MODEL_JSON, FEATURES_JSON, INFERENCE_JSON, GOLDEN_JSON):
        rp = p.resolve()
        assert rental_api not in rp.parents, f"{rp} must NOT live under the rental chrome-extension/api/"
    assert SALE_API_DIR.resolve().name == "sale_api"
    assert (ROOT / "output").resolve() in MODEL_JSON.resolve().parents


def test_inc4_golden_feature_order_matches_features_json():
    """The committed golden's feature_order is byte-equal to features.json (34, frozen)."""
    feature_cols = json.loads(FEATURES_JSON.read_text())
    golden = json.loads(GOLDEN_JSON.read_text())
    assert golden["feature_order"] == feature_cols
    assert golden["n_features"] == len(feature_cols) == 34
