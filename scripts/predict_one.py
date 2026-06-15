#!/usr/bin/env python3
"""
predict_one.py — Single-property fair-value prediction from the CANONICAL model.

Replaces the inline v15-era feature dict in `.github/workflows/predict.yml`. That
inline builder reimplemented ~50 features by hand and loaded rental_model_v15.pkl,
so on-demand predictions used a DIFFERENT model + schema than the served/cached
ones. This script delegates to canonical_predict.predict_one() — the single
linchpin that loads the canonical model and builds features via the canonical
pipeline (which canonical_predict resolves from retrain_canonical.py). No feature
math here; version changes are one edit in retrain_canonical.py.

Input: a JSON object (file via --input, or stdin) with any of:
    bedrooms, bathrooms, size_sqft, postcode, description, features, asking_price,
    property_type, address, latitude, longitude
Missing size is estimated from chrome-extension/api/size_lookup.json (postcode_beds).

Output: prediction_result.json (and stdout) with:
    {asking_price, fair_value, range_low, range_high, premium_pct, assessment,
     size_sqft, size_source, amenities_detected}

Run:
    echo '{"bedrooms":2,"size_sqft":800,"postcode":"SW3 1AA","asking_price":4500}' \
        | python3 scripts/predict_one.py
    python3 scripts/predict_one.py --input payload.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the modeler's blessed canonical module (single source of truth for model
# loading + feature building + prediction). Lead's ruling: import, not reimplement.
import canonical_predict as cp  # noqa: E402

SIZE_LOOKUP_PATH = PROJECT_ROOT / "chrome-extension" / "api" / "size_lookup.json"
OUT_PATH = PROJECT_ROOT / "prediction_result.json"

AMENITY_KEYWORDS = {
    "balcony": "balcony", "terrace": "terrace", "garden": "garden",
    "porter": "porter", "concierge": "porter", "gym": "gym", "pool": "pool",
    "parking": "parking", "lift": "lift", "view": "view",
}


def detect_amenities(text: str) -> list[str]:
    text = (text or "").lower()
    found = []
    for kw, label in AMENITY_KEYWORDS.items():
        if kw in text and label not in found:
            found.append(label)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Path to JSON payload (else stdin)")
    args = ap.parse_args()

    raw = Path(args.input).read_text() if args.input else sys.stdin.read()
    data = json.loads(raw)

    beds = data.get("bedrooms") or 1
    baths = data.get("bathrooms") or 1
    size_sqft = data.get("size_sqft")
    postcode = data.get("postcode") or "SW3 1AA"
    description = data.get("description", "")
    features_txt = data.get("features", "")
    asking_price = data.get("asking_price", 0) or 0
    postcode_district = postcode.split()[0] if postcode else "SW3"

    # Estimate size if missing (same source the extension uses).
    size_source = "provided"
    if not size_sqft:
        size_source = "estimated"
        size_sqft = beds * 400
        if SIZE_LOOKUP_PATH.exists():
            try:
                lookup = json.loads(SIZE_LOOKUP_PATH.read_text())
                size_sqft = lookup.get(f"{postcode_district}_{beds}", size_sqft)
            except Exception:
                pass

    # Delegate the actual model prediction to the modeler's blessed entrypoint
    # (canonical_predict.predict_one). It loads the canonical model and builds
    # features via the canonical pipeline — NO inline feature math here. This script
    # keeps only the I/O + result-schema contract the predict.yml workflow expects.
    fair_value = int(cp.predict_one(
        bedrooms=beds, bathrooms=baths, size_sqft=size_sqft,
        postcode=postcode, postcode_normalized=postcode,
        area=data.get("area", ""),
        property_type=data.get("property_type", "flat"),
        property_type_std=data.get("property_type_std", data.get("property_type", "flat")),
        address=data.get("address", ""),
        latitude=data.get("latitude", 0), longitude=data.get("longitude", 0),
        features=features_txt, description=description,
        source=data.get("source", "rightmove"),
        agent_brand=data.get("agent_brand", data.get("agent_name", "unknown")),
    ))

    if asking_price > 0 and fair_value > 0:
        premium_pct = round((asking_price / fair_value - 1) * 100, 1)
        assessment = "overpriced" if premium_pct > 15 else "good_deal" if premium_pct < -10 else "fair"
    else:
        premium_pct = 0
        assessment = "unknown"

    result = {
        "asking_price": asking_price,
        "fair_value": fair_value,
        "range_low": int(fair_value * 0.79),
        "range_high": int(fair_value * 1.21),
        "premium_pct": premium_pct,
        "assessment": assessment,
        "size_sqft": size_sqft,
        "size_source": size_source,
        "amenities_detected": detect_amenities(f"{description} {features_txt}"),
    }

    print(json.dumps(result, indent=2))
    OUT_PATH.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
