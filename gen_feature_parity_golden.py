"""
gen_feature_parity_golden.py — emit a Python-side GOLDEN FIXTURE for JS↔Python
feature-parity testing of the canonical v20 feature engineering.

Context: a server-side /api/predict (task #28/#29) and the Chrome extension will
share a JS buildFeatures(). This fixture lets artifacts diff their JS output
against the ground-truth Python features key-by-key.

For each sample property we record:
  - inputs: the RAW fields buildFeatures(data) receives (so the JS side feeds the
    SAME inputs)
  - features: the exact {name: value} the canonical pipeline
    (canonical_predict.build_features -> engineer_features_v20) produces, in the
    135-col training order
  - prediction: canonical_predict.predict(...) £ pcm (sanity anchor)

Samples deliberately span the v20 feature-engineering branches that are easy to
get subtly wrong in a JS reimplementation: mews, ultra-prime / prestige street,
social-housing signal, tiny/huge size, premium agent, multi-floor, furnished
keywords, high bathroom count.

Usage:  python3 gen_feature_parity_golden.py   ->  output/feature_parity_golden.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import canonical_predict as cp

OUT = Path('output/feature_parity_golden.json')

# Raw input rows — the fields a /api/predict request (and JS buildFeatures) would
# carry. Cover the tricky branches, not just the happy path.
SAMPLES = [
    {
        '_label': 'belgravia_duplex_premium',  # ultra-prime street + premium agent + multi-floor + furnished
        'bedrooms': 2, 'bathrooms': 2, 'size_sqft': 1312,
        'postcode': 'SW1W 9JA', 'postcode_normalized': 'SW1W', 'area': 'Belgravia',
        'property_type': 'flat', 'property_type_std': 'flat',
        'address': '4 South Eaton Place', 'source': 'savills', 'agent_brand': 'Savills',
        'latitude': 51.4934, 'longitude': -0.1508, 'let_type': 'long',
        'features': '', 'description': 'period duplex, air conditioning, lift, porter, balcony, furnished',
        'floor_count': 2, 'has_first_floor': 1, 'has_second_floor': 1,
    },
    {
        '_label': 'chelsea_flat_plain',  # vanilla flat, rightmove, no amenities
        'bedrooms': 2, 'bathrooms': 1, 'size_sqft': 1000,
        'postcode': 'SW3 2AB', 'postcode_normalized': 'SW3', 'area': 'Chelsea',
        'property_type': 'flat', 'property_type_std': 'flat',
        'address': '12 Some Ordinary Street', 'source': 'rightmove', 'agent_brand': 'unknown',
        'latitude': 51.49, 'longitude': -0.168, 'let_type': 'long',
        'features': '', 'description': 'unfurnished two bed flat',
        'floor_count': 1,
    },
    # ── NAMED SANITY PROPERTIES (the prose validation cases, pinned to exact inputs) ──
    {
        '_label': 'SANITY_sw3_2bed_1000',  # the canonical "SW3 2bed/1000sqft" sanity case
        'bedrooms': 2, 'bathrooms': 1, 'size_sqft': 1000,
        'postcode': 'SW3', 'postcode_normalized': 'SW3', 'area': 'Chelsea',
        'property_type': 'flat', 'property_type_std': 'flat',
        'address': '', 'source': 'rightmove', 'agent_brand': 'unknown',
        'latitude': 51.49, 'longitude': -0.168, 'let_type': 'long',
        'features': '', 'description': '',
        'floor_count': 0,
    },
    {
        '_label': 'SANITY_4_south_eaton_place',  # the canonical "4 South Eaton Place" sanity case
        'bedrooms': 2, 'bathrooms': 2, 'size_sqft': 1312,
        'postcode': 'SW1W', 'postcode_normalized': 'SW1W', 'area': 'Belgravia',
        'property_type': 'flat', 'property_type_std': 'flat',
        'address': '4 South Eaton Place', 'source': 'savills', 'agent_brand': 'Savills',
        'latitude': 51.4934, 'longitude': -0.1508, 'let_type': 'long',
        'features': '', 'description': '',
        'floor_count': 0, 'has_first_floor': 1,
    },
    {
        '_label': 'mews_house',  # is_mews branch (excluded from is_house)
        'bedrooms': 3, 'bathrooms': 2, 'size_sqft': 1600,
        'postcode': 'SW7 1AA', 'postcode_normalized': 'SW7', 'area': 'South Kensington',
        'property_type': 'mews house', 'property_type_std': 'mews house',
        'address': '5 Some Mews', 'source': 'knightfrank', 'agent_brand': 'Knight Frank',
        'latitude': 51.495, 'longitude': -0.178, 'let_type': 'long',
        'features': '', 'description': 'charming mews house with garage',
        'floor_count': 2,
    },
    {
        '_label': 'tiny_studio',  # is_tiny + is_studio branch
        'bedrooms': 1, 'bathrooms': 1, 'size_sqft': 320,
        'postcode': 'W2 1AA', 'postcode_normalized': 'W2', 'area': 'Paddington',
        'property_type': 'studio', 'property_type_std': 'studio',
        'address': 'Flat 9, A Building', 'source': 'foxtons', 'agent_brand': 'unknown',
        'latitude': 51.515, 'longitude': -0.178, 'let_type': 'long',
        'features': '', 'description': 'compact studio',
        'floor_count': 1,
    },
    {
        '_label': 'huge_high_bath_prestige',  # is_huge + high_bathroom_count + prestige street
        'bedrooms': 5, 'bathrooms': 5, 'size_sqft': 3500,
        'postcode': 'SW1X 8AA', 'postcode_normalized': 'SW1X', 'area': 'Knightsbridge',
        'property_type': 'house', 'property_type_std': 'house',
        'address': '20 Cadogan Square', 'source': 'savills', 'agent_brand': 'Savills',
        'latitude': 51.498, 'longitude': -0.16, 'let_type': 'long',
        'features': '', 'description': 'magnificent house, gym, pool, lift, air conditioning, roof terrace',
        'floor_count': 4, 'has_first_floor': 1, 'has_second_floor': 1, 'has_third_floor': 1,
    },
]

REQUIRED_INPUT_FIELDS = [
    'bedrooms', 'bathrooms', 'size_sqft', 'postcode', 'postcode_normalized', 'area',
    'property_type', 'property_type_std', 'address', 'source', 'agent_brand',
    'latitude', 'longitude', 'let_type', 'features', 'description',
    'floor_count', 'has_basement', 'has_ground', 'has_first_floor',
    'has_second_floor', 'has_third_floor', 'has_fourth_plus', 'has_roof_terrace',
]


def main():
    model, feat_order = cp.load_canonical()
    out = {
        'canonical_version': cp.CANONICAL_VERSION,
        'n_features': len(feat_order),
        'feature_order': list(feat_order),   # THE 135-col order; JS must emit these names
        'required_input_fields': REQUIRED_INPUT_FIELDS,
        'note': ('For each sample: feed `inputs` to JS buildFeatures(), then compare its '
                 'output dict to `features` key-by-key (abs diff > 1e-6 = mismatch). '
                 'Order is enforced separately by features.json; this checks VALUES.'),
        'samples': [],
    }

    # engineer_features_v20 reads the floor flags as existing columns (df['has_basement']
    # .fillna(0), not df.get(...)), so any floor flag a request omits must default to 0
    # BEFORE build_features. The JS buildFeatures + /api/predict request layer must do the
    # same (treat absent floor flags as 0). We normalise inputs here so the golden vector
    # reflects that contract.
    FLOOR_DEFAULTS = {
        'has_basement': 0, 'has_ground': 0, 'has_first_floor': 0, 'has_second_floor': 0,
        'has_third_floor': 0, 'has_fourth_plus': 0, 'has_roof_terrace': 0, 'floor_count': 0,
    }

    for s in SAMPLES:
        label = s['_label']
        row = {k: v for k, v in s.items() if not k.startswith('_')}
        for k, dflt in FLOOR_DEFAULTS.items():
            row.setdefault(k, dflt)
        df = pd.DataFrame([row])
        X, cols = cp.build_features(df)
        # Align to the canonical artifact order (same as predict()).
        Xr = X.reindex(columns=feat_order, fill_value=0)
        feats = {name: float(Xr.iloc[0][name]) for name in feat_order}
        pred = float(cp.predict(df)[0])
        out['samples'].append({
            'label': label,
            'inputs': row,
            'prediction_pcm': round(pred, 2),
            'features': feats,
        })
        print(f"  {label:32} pred=£{pred:,.0f}  ({sum(1 for v in feats.values() if v != 0)} non-zero feats)")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}  ({len(out['samples'])} samples, {out['n_features']} features each)")


if __name__ == '__main__':
    main()
