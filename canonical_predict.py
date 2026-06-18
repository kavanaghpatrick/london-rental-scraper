"""
canonical_predict.py — THE single source of truth for the canonical rental
price model (v20). Import this everywhere instead of re-implementing the feature
dict inline.

WHY THIS EXISTS
---------------
Today `automation`, `artifacts`, `.github/workflows/predict.yml` and
`generate-predictions.yml` each re-build the model's feature dict by hand (their
own AMENITY_FEATURES / PRIME_POSTCODES / TUBE_STATIONS / one-hot logic). When the
canonical model changed (v15 → v20) those inline copies silently went out of sync
with the trained feature set → silently WRONG predictions (the model reads the
wrong columns, zero-filled). This module removes the duplication: there is ONE
build_features() and ONE export, both delegating to rental_price_models_v20.py
(the canonical feature engineering).

PUBLIC API
----------
    build_features(df)            -> (X: DataFrame in canonical training order,
                                       feature_cols: list[str])
    load_canonical()              -> (model, feature_cols)   # cached
    predict(df)                   -> np.ndarray of £ pcm  (handles log1p/expm1)
    predict_one(**kwargs)         -> float  (convenience for a single property)
    export_to_chrome(out_dir=...) -> (model_json_path, features_json_path)  # ONE matched pair
    retrain(db_path, save=True)   -> dict  (retrains canonical, writes artifacts)

CANONICAL ARTIFACTS (deterministic paths — automation + serving load THESE):
    output/rental_model_canonical.pkl
    output/rental_model_canonical_features.pkl
    output/rental_model_canonical_meta.json

The canonical model IS v20. The version is declared in ONE place —
`retrain_canonical.py` (`CANON_VERSION` + the `rental_price_models_v<NN>` it
imports). This module DERIVES its feature-engineering bindings from there via
`scripts/_canonical_features.py`, so it can never name a different version than
the trainer/serving side. To switch versions, change CANON_VERSION (and the model
import) in retrain_canonical.py — this module, generate_predictions.py and
predict_one.py all follow automatically.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Canonical version wiring (SINGLE SOURCE = retrain_canonical.CANON_VERSION) ──
# The canonical version is declared in ONE place: retrain_canonical.py
# (its CANON_VERSION constant + the rental_price_models_v<NN> module it imports).
# We DERIVE the feature-engineering bindings from there via the shared resolver
# scripts/_canonical_features.py, so this module can never declare a different
# version than the trainer/serving side. Flip CANON_VERSION in retrain_canonical.py
# and everyone — trainer, this module, generate_predictions.py, predict_one.py —
# follows automatically. (This is the final lock on "one version, everywhere".)
import sys as _sys
from pathlib import Path as _Path
_SCRIPTS = _Path(__file__).resolve().parent / 'scripts'
if str(_SCRIPTS) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS))
from _canonical_features import resolve_feature_pipeline as _resolve_fe, resolve_model_module as _resolve_mod

_ENGINEER, _FEATURE_COLS, CANONICAL_VERSION = _resolve_fe()  # df->df, df->list[str], version
_canon = _resolve_mod()                                       # the rental_price_models_v<NN> module
_BUILD_XGB = _canon.build_xgboost

# ── Paths ───────────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent / 'output'
MODEL_PATH = OUT / 'rental_model_canonical.pkl'
FEATURES_PATH = OUT / 'rental_model_canonical_features.pkl'
META_PATH = OUT / 'rental_model_canonical_meta.json'
INFERENCE_STATS_PATH = OUT / 'rental_model_canonical_inference.json'

_CACHE: dict = {'model': None, 'features': None, 'inference': None}


def load_inference_stats():
    """Load the baked training statistics single-row inference must inject
    (frequency-encoding maps + defaults). Cached. Returns {} if absent (then
    inference falls back to the raw — possibly degenerate — engineered values).
    Generate via gen_inference_stats.py.
    """
    if _CACHE['inference'] is not None:
        return _CACHE['inference']
    if INFERENCE_STATS_PATH.exists():
        with open(INFERENCE_STATS_PATH) as f:
            _CACHE['inference'] = json.load(f)
    else:
        _CACHE['inference'] = {}
    return _CACHE['inference']


# ── Feature construction (the linchpin) ─────────────────────────────────────
def build_features(df: pd.DataFrame, inference: bool = True):
    """Build the canonical feature matrix from a raw listings-shaped DataFrame.

    `df` must have the columns the v20 engineering reads (bedrooms, bathrooms,
    size_sqft, postcode/postcode_normalized, address, property_type_std, source,
    agent_brand, latitude, longitude, features, description, floor flags, …).
    Missing optional columns are tolerated — they fill to safe defaults.

    INFERENCE MODE (default): frequency-encoding features (postcode_freq,
    postcode_area_freq) are computed by v20 FE as (count in THIS frame / len) — a
    single-row request makes them degenerate to 1.0, mismatching the trained model.
    With `inference=True` we OVERRIDE them from the baked training maps
    (load_inference_stats / gen_inference_stats.py) so a single-property estimate
    matches how the model was trained. Pass `inference=False` only when df IS a
    full training-distribution frame (e.g. internal training/eval). See task #30.

    Returns (X, feature_cols) where X is numeric, NaN-free, columns in the EXACT
    training order. Feed X (or X[feature_cols]) straight to the model.
    """
    df = df.copy()

    # Tolerate frames that didn't pre-compute helper cols the engineer expects.
    if 'ppsf' not in df.columns and {'price_pcm', 'size_sqft'} <= set(df.columns):
        df['ppsf'] = df['price_pcm'] / df['size_sqft'].replace(0, np.nan)
    if 'postcode_district' not in df.columns:
        src = df['postcode_normalized'] if 'postcode_normalized' in df.columns else df.get('postcode')
        if src is not None:
            df['postcode_district'] = src.astype(str).str.extract(r'^([A-Z]+\d+[A-Z]?)', expand=False)

    engineered = _ENGINEER(df)
    feature_cols = _FEATURE_COLS(engineered)

    # Zero-fill any declared-but-absent column (parity with how each script trains)
    for c in feature_cols:
        if c not in engineered.columns:
            engineered[c] = 0

    # ── Inference-time frequency override ────────────────────────────────────
    # The model was fit with training-set frequencies; injecting them here fixes
    # the single-row degeneration (postcode_freq/area_freq → 1.0).
    if inference:
        stats = load_inference_stats()
        if stats:
            pcd = engineered.get('postcode_district')
            pca = engineered.get('postcode_area')
            akey = engineered.get('area_key')
            if pcd is not None and 'postcode_freq' in feature_cols:
                fmap = stats.get('postcode_freq', {})
                fdef = stats.get('postcode_freq_default', 0.0)
                engineered['postcode_freq'] = pcd.astype(str).map(lambda k: fmap.get(k, fdef))
            if pca is not None and 'postcode_area_freq' in feature_cols:
                amap = stats.get('postcode_area_freq', {})
                adef = stats.get('postcode_area_freq_default', 0.0)
                engineered['postcode_area_freq'] = pca.astype(str).map(lambda k: amap.get(k, adef))
            # FIX 2: neighborhood `area` overrides. The engineered frame's area_freq is a
            # degenerate per-frame recompute and area_te is a global-prior placeholder
            # (no target at inference); inject the baked training maps so a single-property
            # estimate matches how the model was trained.
            if akey is not None and 'area_freq' in feature_cols:
                afmap = stats.get('area_freq', {})
                afdef = stats.get('area_freq_default', 0.0)
                engineered['area_freq'] = akey.astype(str).map(lambda k: afmap.get(k, afdef))
            if akey is not None and 'area_te' in feature_cols:
                temap = stats.get('area_target_encoding', {})
                tedef = stats.get('area_te_default', 0.0)
                engineered['area_te'] = akey.astype(str).map(lambda k: temap.get(k, tedef))

    X = engineered[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == 'bool':
            X[c] = X[c].astype(int)
        elif X[c].dtype == 'object':
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)
    return X, feature_cols


# ── Model loading + prediction ──────────────────────────────────────────────
def load_canonical():
    """Load the canonical model + feature order (cached). Raises if missing."""
    if _CACHE['model'] is not None:
        return _CACHE['model'], _CACHE['features']
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Canonical artifacts missing ({MODEL_PATH}, {FEATURES_PATH}). "
            f"Run canonical_predict.retrain(db_path) or retrain_canonical.py first."
        )
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        features = pickle.load(f)
    if model.n_features_in_ != len(features):
        raise ValueError(
            f"Artifact mismatch: model expects {model.n_features_in_} features "
            f"but features pkl has {len(features)}. Re-run retrain."
        )
    _CACHE['model'], _CACHE['features'] = model, features
    return model, features


def predict(df: pd.DataFrame) -> np.ndarray:
    """Predict £ pcm for each row of a listings-shaped DataFrame.

    Builds features via build_features() (so the column set/order can NEVER drift
    from training), aligns to the artifact's feature order, applies the model and
    inverts the log1p target with expm1.
    """
    model, feat_order = load_canonical()
    X, _ = build_features(df)
    # Align to the artifact's exact order; zero-fill any the model wants but FE
    # didn't emit for this batch (e.g. a property-type dummy unseen in this slice).
    X = X.reindex(columns=feat_order, fill_value=0)
    return np.expm1(model.predict(X))


def predict_one(**kwargs) -> float:
    """Convenience: predict a single property from keyword fields.

    Accepts the same column names build_features reads. Unknown/missing fields
    default safely. Returns £ pcm (float).
    """
    row = {
        'bedrooms': kwargs.get('bedrooms', 1),
        'bathrooms': kwargs.get('bathrooms', 1),
        'size_sqft': kwargs.get('size_sqft', 0),
        'postcode': kwargs.get('postcode', ''),
        'postcode_normalized': kwargs.get('postcode_normalized', kwargs.get('postcode', '')),
        'area': kwargs.get('area', ''),
        'property_type': kwargs.get('property_type', 'flat'),
        'property_type_std': kwargs.get('property_type_std', kwargs.get('property_type', 'flat')),
        'address': kwargs.get('address', ''),
        'price_pcm': kwargs.get('price_pcm', 0),  # only used for ppsf helper; not a feature
        'latitude': kwargs.get('latitude', 0),
        'longitude': kwargs.get('longitude', 0),
        'features': kwargs.get('features', ''),
        # FIX 3: the real free text lives in `summary` (description is ~empty in the DB).
        # Pass it through so single-row inference is not text-blind vs training, which now
        # drives all text extractors off COALESCE(summary, description). Defaults to '' so
        # callers that omit it still work (engineer_features_v20 tolerates a missing column).
        'summary': kwargs.get('summary', ''),
        'description': kwargs.get('description', ''),
        'source': kwargs.get('source', 'rightmove'),
        'let_type': kwargs.get('let_type', 'long'),
        'agent_brand': kwargs.get('agent_brand', kwargs.get('agent_name', 'unknown')),
        'postcode_inferred': kwargs.get('postcode_inferred', 0),
        'floor_count': kwargs.get('floor_count', 0),
        'has_roof_terrace': kwargs.get('has_roof_terrace', 0),
        'has_basement': kwargs.get('has_basement', 0),
        'has_ground': kwargs.get('has_ground', 0),
        'has_first_floor': kwargs.get('has_first_floor', 0),
        'has_second_floor': kwargs.get('has_second_floor', 0),
        'has_third_floor': kwargs.get('has_third_floor', 0),
        'has_fourth_plus': kwargs.get('has_fourth_plus', 0),
    }
    return float(predict(pd.DataFrame([row]))[0])


# ── Export (ONE matched model.json + features.json) ─────────────────────────
def export_to_chrome(out_dir: str = 'chrome-extension/api'):
    """Emit model.json + features.json as ONE matched pair for the extension.

    Loads the canonical model + the EXACT feature order it was trained on and
    writes both from that single artifact, so the JSON model and the JSON feature
    list can never drift from each other.

    NOTE: we serialize via the underlying Booster (`get_booster().save_model`)
    rather than the sklearn wrapper's `save_model`, because a *unpickled*
    XGBRegressor can lose its `_estimator_type` metadata under xgboost>=2 and
    `XGBRegressor.save_model` then raises. The Booster JSON is exactly what
    chrome-extension/xgboost.js parses (it reads learner.gradient_booster...),
    so this is the correct, robust path.
    """
    model, features = load_canonical()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / 'model.json'
    model.get_booster().save_model(str(model_path))

    features_path = out / 'features.json'
    with open(features_path, 'w') as f:
        json.dump(list(features), f, indent=2)

    # Hard guarantee the pair matches the canonical artifact.
    with open(features_path) as f:
        exported = json.load(f)
    assert exported == list(features), "features.json does not match canonical feature order"
    print(f"[export_to_chrome] {model_path} + {features_path} "
          f"({len(features)} features, matched pair)")
    return model_path, features_path


# ── Retrain (delegates to the canonical pipeline) ───────────────────────────
def retrain(db_path: str = 'output/rentals.db', save: bool = True) -> dict:
    """Retrain the canonical model on `db_path` and (optionally) save artifacts.

    Thin wrapper around retrain_canonical.py so automation has ONE entrypoint
    callable either as a module function or a script.
    """
    import retrain_canonical as rc
    import sys
    argv = sys.argv
    sys.argv = ['retrain_canonical.py', '--db', db_path]
    try:
        meta = rc.main()
    finally:
        sys.argv = argv
    # Invalidate cache so the next predict() picks up the fresh model.
    _CACHE['model'] = _CACHE['features'] = None
    return meta


if __name__ == '__main__':
    # Self-test: load artifacts, build features for one property, predict, export-dry-check.
    m, feats = load_canonical()
    print(f"Canonical {CANONICAL_VERSION}: model expects {m.n_features_in_} feats, "
          f"features pkl has {len(feats)} -> {'OK' if m.n_features_in_ == len(feats) else 'MISMATCH'}")
    est = predict_one(size_sqft=1312, bedrooms=2, bathrooms=2, postcode='SW1W',
                      postcode_normalized='SW1W', area='Belgravia', property_type='flat',
                      address='4 South Eaton Place', source='savills', agent_brand='Savills',
                      latitude=51.4934, longitude=-0.1508)
    print(f"predict_one(4 South Eaton Place, 1312sqft, 2bed): £{est:,.0f} pcm")
