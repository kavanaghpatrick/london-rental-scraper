#!/usr/bin/env python3
"""
Export the CANONICAL model (output/rental_model_canonical.pkl) to the Chrome
extension artifact format, with ZERO drift from the served model.

Why this and not `rental_price_models_v20.py --export`:
  `--export` RE-TRAINS a fresh booster on the `is_active=1` (or Postgres) frame,
  which is a DIFFERENT row set than the canonical recency-independent retrain.
  Same hyperparameters, different data -> a different booster -> drift. Exporting
  the canonical pkl's booster directly guarantees chrome-extension/api/model.json
  IS the canonical model, byte-for-byte.

Loader contract (chrome-extension/xgboost.js):
  - model.json  : native XGBoost JSON (booster.save_model). Reads
                  learner.gradient_booster.model.trees[] and base_score.
  - features.json: a FLAT, ORDER-SENSITIVE list[str]. The loader builds the
                  feature vector from THIS order and zero-fills any name the
                  extension's buildFeatures() does not emit. The canonical
                  135-feature order (rental_model_canonical_features.pkl) IS the
                  contract.

Outputs (overwrites in place):
  chrome-extension/api/model.json
  chrome-extension/api/features.json
"""

import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
API_DIR = ROOT / 'chrome-extension' / 'api'

CANONICAL_MODEL = OUTPUT_DIR / 'rental_model_canonical.pkl'
CANONICAL_FEATURES = OUTPUT_DIR / 'rental_model_canonical_features.pkl'

MODEL_JSON = API_DIR / 'model.json'
FEATURES_JSON = API_DIR / 'features.json'


def export():
    if not CANONICAL_MODEL.exists() or not CANONICAL_FEATURES.exists():
        sys.exit(f"ERROR: canonical artifacts missing under {OUTPUT_DIR}/ "
                 f"(need rental_model_canonical.pkl + _features.pkl)")

    model = pickle.load(open(CANONICAL_MODEL, 'rb'))
    feature_cols = list(pickle.load(open(CANONICAL_FEATURES, 'rb')))

    n_in = getattr(model, 'n_features_in_', None)
    if n_in != len(feature_cols):
        sys.exit(f"ERROR: model.n_features_in_={n_in} != len(features)={len(feature_cols)}")

    API_DIR.mkdir(parents=True, exist_ok=True)

    # Native XGBoost JSON straight from the canonical booster (no retrain).
    booster = model.get_booster()
    booster.save_model(str(MODEL_JSON))

    # Feature order = canonical contract.
    with open(FEATURES_JSON, 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Wrote {MODEL_JSON}")
    print(f"Wrote {FEATURES_JSON}")
    print(f"  features: {len(feature_cols)}  trees: {model.n_estimators}")
    return MODEL_JSON, FEATURES_JSON


if __name__ == '__main__':
    export()
