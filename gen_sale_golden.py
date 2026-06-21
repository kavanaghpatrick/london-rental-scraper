"""gen_sale_golden.py — re-score the COMMITTED for-sale Booster into the parity golden.

Inc4a thin wrapper (the for-sale analogue of gen_feature_parity_golden.py). It does NOT
retrain: it LOADS the committed output/sale_api/model.json as an xgboost Booster, the
committed features.json (the 34-name order), and the committed sale_model_inference.json
(the baked district_freq / postcode_area_freq maps + numeric defaults), then scores the
deterministic _GOLDEN_INPUTS from for_sale.sale_retrain with build_features(inference=True)
— the EXACT path the JS serving predictor and predict_one_default use — and writes
output/sale_feature_parity_golden.json (key `prediction_price`, a £ lump sum, NOT
`prediction_pcm`).

WHY load+score-the-committed-model rather than retrain (spec section 1):
  A retrain in CI could yield a Booster whose float predictions differ from the committed
  model.json across xgboost minor versions, drifting the golden from the model the /api
  route actually serves. Scoring the COMMITTED Booster eliminates that class entirely — the
  gate compares JS predictions against the SAME bytes the route ships.

BLOCKER 1: features are built with inference=True using the baked maps (NOT the degenerate
single-row inference=False recompute, which collapses district_freq / postcode_area_freq to
1.0 and diverges from serving). This is the SINGLE correct golden writer alongside
sale_retrain.gen_sale_feature_parity_golden — both build inference=True from the same maps.

Network-free, deterministic, no RNG. Usage: python3 gen_sale_golden.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from for_sale import sale_price_model
from for_sale.sale_retrain import (
    REQUIRED_INPUT_FIELDS,
    SALE_CANON_VERSION,
    _GOLDEN_INPUTS,
)

ROOT = Path(__file__).resolve().parent
MODEL_JSON = ROOT / "output" / "sale_api" / "model.json"
FEATURES_JSON = ROOT / "output" / "sale_api" / "features.json"
INFERENCE_JSON = ROOT / "output" / "sale_model_inference.json"
GOLDEN_OUT = ROOT / "output" / "sale_feature_parity_golden.json"


def build_golden() -> dict:
    """Load the committed Booster + features + baked maps, score the golden inputs with
    inference=True, and return the golden dict (schema-identical to the Inc3 writer)."""
    # The COMMITTED Booster — loaded, never retrained (byte-safety guarantee).
    booster = xgb.Booster()
    booster.load_model(str(MODEL_JSON))

    feature_cols = json.loads(FEATURES_JSON.read_text())
    assert isinstance(feature_cols, list) and feature_cols, "features.json must be a non-empty list"

    inf = json.loads(INFERENCE_JSON.read_text())
    freq_map = inf["district_freq"]
    freq_default = float(inf["district_freq_default"])
    area_freq_map = inf["postcode_area_freq"]
    area_freq_default = float(inf["postcode_area_freq_default"])

    samples = []
    for inp in _GOLDEN_INPUTS:
        row = {k: v for k, v in inp.items() if k != "label"}
        # inference=True with the BAKED maps — the serving path (BLOCKER 1).
        X, _ = sale_price_model.build_features(
            [row],
            inference=True,
            freq_map=freq_map,
            freq_default=freq_default,
            area_freq_map=area_freq_map,
            area_freq_default=area_freq_default,
        )
        X = X.reindex(columns=feature_cols, fill_value=0.0).astype(float)
        # Score with the loaded Booster via DMatrix (feature_names aligned to the order).
        dmat = xgb.DMatrix(X.to_numpy(dtype=float), feature_names=list(feature_cols))
        pred_log = booster.predict(dmat)
        price = float(np.expm1(pred_log)[0])
        feat_values = {c: float(X.iloc[0][c]) for c in feature_cols}
        samples.append({
            "label": inp["label"],
            "inputs": row,
            "prediction_price": price,
            "features": feat_values,
        })

    return {
        "canonical_version": SALE_CANON_VERSION,
        "n_features": len(feature_cols),
        "feature_order": list(feature_cols),
        "required_input_fields": list(REQUIRED_INPUT_FIELDS),
        "samples": samples,
    }


def main() -> None:
    golden = build_golden()
    GOLDEN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_OUT, "w") as f:
        json.dump(golden, f, indent=2)
    n = len(golden["samples"])
    print(
        f"[gen_sale_golden] wrote {GOLDEN_OUT} — {n} samples, "
        f"{golden['n_features']} features each (scored the COMMITTED Booster, inference=True)"
    )


if __name__ == "__main__":
    main()
