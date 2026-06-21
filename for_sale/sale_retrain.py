"""for_sale.sale_retrain — the ISOLATED training harness + artifact generators for the
FOR-SALE sale-price model (Inc3).

It is the for-sale analogue of retrain_canonical.py / gen_inference_stats.py /
canonical_predict.export_to_chrome — but a SEPARATE module family that imports ONLY
for_sale.sale_price_model + for_sale.sale_data + stdlib/numpy/pandas/sklearn/xgboost.

ISOLATION CONTRACT (enforced by the guard tests in tests/test_for_sale_inc3_model.py)
-------------------------------------------------------------------------------------
  * NEVER imports the rental MODEL chain: rental_price_models_v20 / canonical_predict /
    retrain_canonical, nor the rental sidecar generators (gen_inference_stats /
    gen_feature_parity_golden / sync_js_freq_maps). Every rental PATTERN below — the
    artifact+meta layout, the freq-default invariant (_assert_default_shape), the
    Booster-JSON export, the parity golden — is RE-IMPLEMENTED BY VALUE so a rental
    retrain can never perturb the sale model and vice-versa.
  * All artifacts are sale-named and live under output/ (sale_api/ NOT chrome-extension/
    api/, which is the rental dir).
  * NETWORK-FREE + DETERMINISTIC: seed=42, XGBRegressor n_jobs=1 (in sale_price_model.train),
    trained on the committed tests/fixtures/for_sale/sale_training_sample.json when
    output/sales.db is absent (the CI path).

The artifacts are REGENERABLE — run_sale_retrain() produces them deterministically from
the committed fixture, so a clean checkout can rebuild the entire Inc4-unblocking bundle
network-free. Tests redirect every path to tmp_path; this module never assumes the repo's
output/ is writable.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from for_sale import sale_data, sale_price_model

# Re-export train() so the harness presents a single retrain entrypoint surface
# (sale_retrain.train / sale_retrain.run_sale_retrain). The implementation lives in
# sale_price_model (Writer1's monotone + return-dict edit); this is a name re-export, NOT a
# second copy — there is exactly one train() in the for-sale vertical.
train = sale_price_model.train

# ─────────────────────────────────────────────────────────────────────────────────────
# 2.1 CONSTANTS / PATHS
#
# These are MODULE-LEVEL globals on purpose: the test harness monkeypatches them on this
# module (test_for_sale_inc3_model._redirect_outputs sets OUTPUT_DIR / MODEL_PATH / … to
# tmp_path). Every writer below reads them from the module namespace AT CALL TIME (via the
# _paths() helper) so the redirect is honoured and the repo's output/ is never polluted.
# ─────────────────────────────────────────────────────────────────────────────────────

SALE_CANON_VERSION = "sale_v1"
SEED = 42

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE = _ROOT / "tests" / "fixtures" / "for_sale" / "sale_training_sample.json"
SALES_DB = _ROOT / "output" / "sales.db"

# Reuse the sale_price_model output dir so the pickle pair lands beside the sidecars. All
# sale-named, never the rental artifact / chrome-extension/api dir.
OUTPUT_DIR = sale_price_model.OUTPUT_DIR
MODEL_PATH = OUTPUT_DIR / "sale_model.pkl"
FEATURES_PATH = OUTPUT_DIR / "sale_model_features.pkl"
META_PATH = OUTPUT_DIR / "sale_model_meta.json"
INFERENCE_PATH = OUTPUT_DIR / "sale_model_inference.json"
SALE_API_DIR = OUTPUT_DIR / "sale_api"          # NEVER chrome-extension/api (that is rental)
GOLDEN_PATH = OUTPUT_DIR / "sale_feature_parity_golden.json"

# The exact input fields build_features reads — the Inc4 request-layer contract (documented
# here + echoed in the golden's required_input_fields).
REQUIRED_INPUT_FIELDS = (
    "postcode",
    "bedrooms",
    "bathrooms",
    "size_sqft",
    "property_type",
    "address",
    "is_new_build",
    "latitude",
    "longitude",
    "price_qualifier",
)


def _paths() -> dict[str, Path]:
    """Resolve the artifact paths from THIS module's globals at call time.

    The test harness monkeypatches MODEL_PATH / META_PATH / … on the module; reading them
    here (rather than capturing them as default arguments at import) is what lets the
    redirect-to-tmp_path actually take effect. Returns a dict of resolved Paths.
    """
    import for_sale.sale_retrain as _self  # this module, post-monkeypatch

    return {
        "model": Path(_self.MODEL_PATH),
        "features": Path(_self.FEATURES_PATH),
        "meta": Path(_self.META_PATH),
        "inference": Path(_self.INFERENCE_PATH),
        "api_dir": Path(_self.SALE_API_DIR),
        "golden": Path(_self.GOLDEN_PATH),
    }


# ─────────────────────────────────────────────────────────────────────────────────────
# 2.2 ROW LOADER
# ─────────────────────────────────────────────────────────────────────────────────────


def _plausible(row: dict) -> bool:
    """Sale-magnitude / sale-ppsf plausibility filter (mirrors sale_price_model bounds by
    value — a data-quality gate, NOT a feature). A row with no asking_price (POA) is
    dropped here too: it has no label."""
    price = row.get("asking_price")
    if not sale_price_model.is_plausible_sale_price(price):
        return False
    size = row.get("size_sqft")
    if size:
        try:
            ppsf = float(price) / float(size)
        except (TypeError, ValueError, ZeroDivisionError):
            return True  # unparseable size → skip the ppsf gate, keep the row
        if not sale_price_model.is_plausible_sale_ppsf(ppsf):
            return False
    return True


def load_sale_rows(
    db_path: Path | str = SALES_DB,
    sample_path: Path | str = DEFAULT_SAMPLE,
) -> list[dict]:
    """Load the for-sale training rows — network-free, deterministic, no RNG.

    * If `db_path` exists: read via sale_data.fetch_sale_listings(active_only=True), then
      apply the deterministic DATA-LOAD FILTERS (no asking_price → drop; under-offer/SSTC →
      drop; implausible sale price / ppsf → drop) and a stable, RNG-free de-dup by
      (source, property_id) keep-first (belt-and-braces over the table's UNIQUE).
    * Else (the CI path — output/sales.db does not exist): json.loads the committed sample.
      The sample is pre-cleaned (300 rows / zero nulls) but the SAME plausibility filter is
      applied for parity with the DB branch.

    Returns list[dict]. NETWORK-FREE either branch.
    """
    db_path = Path(db_path)
    sample_path = Path(sample_path)

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            raw = sale_data.fetch_sale_listings(conn, active_only=True)
        finally:
            conn.close()
        rows: list[dict] = []
        for r in raw:
            if r.get("asking_price") is None:
                continue  # POA — no label
            if r.get("is_under_offer") == 1:
                continue  # SSTC / under offer — excludable comp
            if not _plausible(r):
                continue
            rows.append(r)
        # Stable RNG-free de-dup by (source, property_id), keep-first. fetch_sale_listings
        # already returns a deterministic ORDER BY id, so iteration order is stable.
        seen: set[tuple] = set()
        deduped: list[dict] = []
        for r in rows:
            key = (r.get("source"), r.get("property_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped

    # CI / fixture path.
    raw = json.loads(sample_path.read_text())
    return [r for r in raw if _plausible(r)]


# ─────────────────────────────────────────────────────────────────────────────────────
# 2.4 MULTI-SEED STABILITY
# ─────────────────────────────────────────────────────────────────────────────────────


def seed_stability(rows: list[dict], seeds=(42, 7, 123)) -> dict:
    """Train once per seed (each fully deterministic) and return the held-out R² spread.

    Used by the structural gate (G7) to prove the model is stable across seeds, not
    seed-lucky. NETWORK-FREE — ~3 quick trains on the committed sample.
    """
    r2_values = [float(sale_price_model.train(rows, seed=s)["metrics"]["r2"]) for s in seeds]
    arr = np.asarray(r2_values, dtype=float)
    return {
        "r2_values": r2_values,
        "r2_std": float(arr.std()),
        "r2_min": float(arr.min()),
    }


# ─────────────────────────────────────────────────────────────────────────────────────
# 5. ARTIFACT GENERATORS (re-implemented by VALUE — never import the rental sidecars)
# ─────────────────────────────────────────────────────────────────────────────────────


def _assert_default_shape(stats: dict) -> None:
    """Fail LOUDLY before writing if the freq-default invariant is violated (mirrors the
    rental gen_inference_stats._assert_default_shape BY VALUE).

    Any downstream reader (the Python predict path AND the Inc4 JS predictor) keys off a
    NUMERIC top-level `*_default` that equals min(map). A None default or a nested in-map
    'default' key is the exact shape that crashed np.log1p / broke JS↔Python parity — never
    write it.
    """
    for base in ("district_freq", "postcode_area_freq"):
        fmap, dflt = stats[base], stats[f"{base}_default"]
        assert isinstance(dflt, float), (
            f"{base}_default must be float, got {type(dflt).__name__}"
        )
        assert dflt == float(min(fmap.values())), (
            f"{base}_default must equal min({base}.values())"
        )
        assert "default" not in fmap, f"{base} map must not embed a nested 'default' key"


def gen_sale_inference_stats(result: dict, inference_path: Path | str | None = None) -> Path:
    """Write the baked freq sidecar (sale_model_inference.json) from a train() result.

    The sidecar carries the district + postcode-area frequency maps (computed on the FULL
    training rows, a leak-free property-attribute distribution) and their NUMERIC defaults
    (== min(map)). It is the linchpin that kills the single-row freq degeneracy for ANY
    downstream predictor (Python OR Inc4 JS). The _assert_default_shape invariant is checked
    BEFORE the file is written, so a malformed sidecar can never reach disk.
    """
    if inference_path is None:
        inference_path = _paths()["inference"]
    inference_path = Path(inference_path)

    stats = {
        "canonical_version": SALE_CANON_VERSION,
        "n_train": int(result["n_train"]),
        "district_freq": {str(k): float(v) for k, v in result["freq_map"].items()},
        "district_freq_default": float(result["freq_default"]),
        "postcode_area_freq": {str(k): float(v) for k, v in result["area_freq_map"].items()},
        "postcode_area_freq_default": float(result["area_freq_default"]),
        "note": (
            "Single-row inference injects district_freq / postcode_area_freq from these "
            "training maps (keyed on postcode_district / postcode_area) instead of the "
            "degenerate per-frame 1.0 recompute. Regenerated atomically with the pkl."
        ),
    }
    _assert_default_shape(stats)

    inference_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inference_path, "w") as f:
        json.dump(stats, f, indent=2)
    return inference_path


def export_sale_to_chrome(model, feature_cols, api_dir: Path | str | None = None):
    """Emit sale_api/model.json (Booster JSON) + sale_api/features.json (the feature order).

    Mirrors canonical_predict.export_to_chrome BY VALUE:
      * model.json is serialized via the underlying Booster (`get_booster().save_model`) — a
        *unpickled* XGBRegressor can lose `_estimator_type` under xgboost>=2 and the sklearn
        wrapper's save_model then raises; the Booster JSON is exactly what a JS tree-walker
        reads (learner.gradient_booster…).
      * features.json is the feature_cols list, and the export ASSERTS exported ==
        list(feature_cols) so the model/feature pair can never silently drift.

    Lands under output/sale_api/ — NEVER chrome-extension/api/ (the rental dir).
    """
    if api_dir is None:
        api_dir = _paths()["api_dir"]
    api_dir = Path(api_dir)
    api_dir.mkdir(parents=True, exist_ok=True)

    model_path = api_dir / "model.json"
    model.get_booster().save_model(str(model_path))

    features_path = api_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump(list(feature_cols), f, indent=2)

    # Hard guarantee the pair matches the in-memory feature order.
    with open(features_path) as f:
        exported = json.load(f)
    assert exported == list(feature_cols), (
        "sale_api/features.json does not match the trained feature order"
    )
    return model_path, features_path


# Golden parity samples — span the easy-to-get-wrong branches (Finding 5): prime vs
# ordinary district, prestige street, house vs flat, penthouse, new-build, missing-postcode
# → UNKNOWN, unseen-district → default-freq, tiny vs huge size, coordless row. Deterministic,
# fixed dicts (no RNG).
_GOLDEN_INPUTS: tuple[dict, ...] = (
    {"label": "prime_flat_sw1x", "postcode": "SW1X", "bedrooms": 3, "bathrooms": 2,
     "size_sqft": 1400, "property_type": "Flat", "address": "Chester Square, London, SW1X"},
    {"label": "ordinary_flat_se15", "postcode": "SE15", "bedrooms": 3, "bathrooms": 2,
     "size_sqft": 1400, "property_type": "Flat", "address": "A Street, London, SE15"},
    {"label": "prestige_street_eaton_square", "postcode": "SW1W", "bedrooms": 4,
     "bathrooms": 3, "size_sqft": 2200, "property_type": "Flat",
     "address": "Eaton Square, London, SW1W"},
    {"label": "house_sw3", "postcode": "SW3", "bedrooms": 5, "bathrooms": 4,
     "size_sqft": 3200, "property_type": "Town House", "address": "A Road, London, SW3"},
    {"label": "penthouse_w1", "postcode": "W1", "bedrooms": 3, "bathrooms": 3,
     "size_sqft": 2000, "property_type": "Penthouse", "address": "A Street, London, W1"},
    {"label": "new_build_sw3", "postcode": "SW3", "bedrooms": 2, "bathrooms": 2,
     "size_sqft": 1000, "property_type": "Flat", "address": "A Street, London, SW3",
     "is_new_build": 1},
    {"label": "missing_postcode_unknown", "postcode": "", "bedrooms": 2, "bathrooms": 1,
     "size_sqft": 800, "property_type": "Flat", "address": "Somewhere, London"},
    {"label": "unseen_district_default_freq", "postcode": "ZZ9", "bedrooms": 2,
     "bathrooms": 1, "size_sqft": 850, "property_type": "Flat",
     "address": "Nowhere, ZZ9 9ZZ"},
    {"label": "tiny_studio", "postcode": "W2", "bedrooms": 1, "bathrooms": 1,
     "size_sqft": 320, "property_type": "Studio", "address": "A Street, London, W2"},
    {"label": "huge_house", "postcode": "NW3", "bedrooms": 6, "bathrooms": 5,
     "size_sqft": 4500, "property_type": "House", "address": "A Road, London, NW3"},
    {"label": "coordless_maisonette", "postcode": "W8", "bedrooms": 3, "bathrooms": 2,
     "size_sqft": 1500, "property_type": "Maisonette", "address": "A Street, London, W8"},
)


def gen_sale_feature_parity_golden(
    model, feature_cols, sample_rows=None, golden_path: Path | str | None = None
):
    """Write sale_feature_parity_golden.json — the Python golden Inc4's sale_fixture_diff.mjs
    diffs key-by-key (mirrors the rental golden BY VALUE; £ lump sum, not pcm).

    Each sample carries its inputs, the predicted £ price, and the FULL engineered feature
    row, so an Inc4 JS port that diverges on any feature OR on the float32 split-arithmetic
    is caught at a TIGHT tolerance. The samples span the easy-to-get-wrong branches.
    `sample_rows` is accepted for signature parity but the golden uses fixed deterministic
    inputs (no dependence on the training fixture order).
    """
    if golden_path is None:
        golden_path = _paths()["golden"]
    golden_path = Path(golden_path)

    samples = []
    for inp in _GOLDEN_INPUTS:
        row = {k: v for k, v in inp.items() if k != "label"}
        X, _ = sale_price_model.build_features([row])
        X = X.reindex(columns=feature_cols, fill_value=0.0).astype(float)
        price = float(np.expm1(model.predict(X))[0])
        feat_values = {c: float(X.iloc[0][c]) for c in feature_cols}
        samples.append({
            "label": inp["label"],
            "inputs": row,
            "prediction_price": price,
            "features": feat_values,
        })

    golden = {
        "canonical_version": SALE_CANON_VERSION,
        "n_features": len(feature_cols),
        "feature_order": list(feature_cols),
        "required_input_fields": list(REQUIRED_INPUT_FIELDS),
        "samples": samples,
    }

    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "w") as f:
        json.dump(golden, f, indent=2)
    return golden_path


# ─────────────────────────────────────────────────────────────────────────────────────
# 2.5 THE FULL HARNESS ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────────────


def run_sale_retrain(
    db_path: Path | str = SALES_DB,
    sample_path: Path | str = DEFAULT_SAMPLE,
    seed: int = SEED,
    *,
    write: bool = True,
) -> dict:
    """Full deterministic retrain entrypoint for the for-sale model (mirrors
    retrain_canonical STRUCTURE, sale-isolated).

    Steps:
      1. rows = load_sale_rows(db_path, sample_path)
      2. result = sale_price_model.train(rows, seed=seed)
      3. if write: save the pickle pair + meta + inference sidecar + sale_api Booster/features
         + the parity golden — all to the (possibly monkeypatched-to-tmp) module paths.
      4. return result.

    db_source in the meta is DERIVED from the actual load source (sales.db when present, else
    the sample fixture) — NOT a hardcoded constant (the Retrain-readiness V2 footgun).
    """
    db_path = Path(db_path)
    sample_path = Path(sample_path)

    rows = load_sale_rows(db_path, sample_path)
    result = sale_price_model.train(rows, seed=seed)

    if not write:
        return result

    paths = _paths()
    model = result["model"]
    feature_cols = result["feature_cols"]

    # a. pickle pair (existing sale_price_model serializer).
    sale_price_model.save_model(model, feature_cols, paths["model"], paths["features"])

    # b. meta sidecar — db_source DERIVED from the actual load source.
    db_source = str(db_path) if db_path.exists() else str(sample_path)
    meta = {
        "canonical_version": SALE_CANON_VERSION,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),  # informational; NEVER asserted
        "db_source": db_source,
        "n_train": int(result["n_train"]),
        "n_test": int(result["n_test"]),
        "n_features": len(feature_cols),
        "target": "log1p(asking_price), inverse expm1",
        "metrics": result["metrics"],
        "seed": int(seed),
        "xgb_params": model.get_params(),
    }
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # c. baked freq sidecar (the BLOCKER-1 fix linchpin).
    gen_sale_inference_stats(result, paths["inference"])

    # d. sale_api Booster JSON + features.json.
    export_sale_to_chrome(model, feature_cols, paths["api_dir"])

    # e. feature-parity golden.
    gen_sale_feature_parity_golden(model, feature_cols, rows, paths["golden"])

    return result


if __name__ == "__main__":  # pragma: no cover — manual regeneration entrypoint
    res = run_sale_retrain(write=True)
    print(
        f"[sale_retrain] trained sale_v1: n_train={res['n_train']} "
        f"n_features={len(res['feature_cols'])} r2={res['metrics']['r2']:.4f}"
    )
