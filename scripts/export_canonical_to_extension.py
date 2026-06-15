#!/usr/bin/env python3
"""
Export the CANONICAL model to the Chrome extension artifact format
(chrome-extension/api/model.json + features.json) with ZERO drift.

This is a thin CLI wrapper around the SINGLE SOURCE OF TRUTH,
`canonical_predict.export_to_chrome()`. All export logic lives there so the
Python serving path, the automation pipeline, and this script can never diverge.

Why delegate (instead of a private implementation):
  - canonical_predict.export_to_chrome() serializes via the underlying Booster
    (`get_booster().save_model`), NOT the sklearn wrapper's save_model. A *unpickled*
    XGBRegressor can lose `_estimator_type` under xgboost>=2, making
    XGBRegressor.save_model raise — which is why `rental_price_models_v20.py --export`
    is broken. The Booster JSON is exactly what chrome-extension/xgboost.js parses
    (learner.gradient_booster.model.trees[] + base_score).
  - It loads the canonical model + the EXACT feature order it was trained on and
    asserts the model.json + features.json pair matches before returning, so the
    JSON model and the JSON feature list can never drift from each other.

Loader contract reminder (chrome-extension/xgboost.js): features.json is a flat,
ORDER-SENSITIVE list; the loader builds the vector from that order and zero-fills
any name buildFeatures() doesn't emit (`?? 0`), so a 135-feature artifact is fully
compatible with the extension's 150-key feature builder.

Usage:
    python3 scripts/export_canonical_to_extension.py [out_dir]
    (out_dir defaults to chrome-extension/api)
"""

import sys
from pathlib import Path

# Make the project root importable so `import canonical_predict` works from scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import canonical_predict as cp  # noqa: E402


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'chrome-extension' / 'api')
    model_path, features_path = cp.export_to_chrome(out_dir)
    print(f"Exported canonical {cp.CANONICAL_VERSION} -> {model_path}, {features_path}")
    return model_path, features_path


if __name__ == '__main__':
    main()
