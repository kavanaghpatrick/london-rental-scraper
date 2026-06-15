"""
gen_inference_stats.py — compute the TRAINING-time statistics that single-row
inference must inject (instead of recomputing from a 1-row frame).

Problem (task #30): engineer_features_v20 computes postcode_freq / postcode_area_freq
as (count in THIS frame / len(frame)). That's correct at TRAIN time (full frame) but
DEGENERATE at single-row inference: one district = 100% of a 1-row "dataset" → freq=1.0,
whereas the model was fit with the real training freq (e.g. SW3≈0.88, W11≈0.023).

This script reproduces the EXACT canonical training frame (same loader/filters as
retrain_canonical) and persists the frequency maps + sensible defaults to
    output/rental_model_canonical_inference.json
canonical_predict.build_features() loads this and overrides the degenerate single-row
freq features so single-property estimates match the trained model.

NOTE: these freqs include the `postcode_normalized.fillna('SW3')` artifact (86.7% of
rows have null postcode_normalized → bucketed as SW3 → SW3 freq≈0.88). The model was
FIT on that, so inference must reproduce it as-is to match — we deliberately do NOT
"correct" it here.

Usage:  python3 gen_inference_stats.py   ->  output/rental_model_canonical_inference.json
"""
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

import retrain_canonical as rc

OUT = Path('output/rental_model_canonical_inference.json')


def main(db='output/rentals.db'):
    df = rc.load_recency_independent(db)
    df = rc.v20.engineer_features_v20(df)
    N = len(df)

    pc_counts = df['postcode_district'].value_counts()
    pc_freq = {str(k): v / N for k, v in pc_counts.items()}
    area_counts = df['postcode_area'].value_counts()
    area_freq = {str(k): v / N for k, v in area_counts.items()}

    stats = {
        'canonical_version': rc.CANON_VERSION,
        'n_train': int(N),
        # Frequency-encoding maps (the model was fit on these exact values).
        'postcode_freq': pc_freq,
        'postcode_area_freq': area_freq,
        # Defaults for unseen keys: smallest observed training freq (a single unseen
        # listing would be ~1/N; min observed is the closest in-distribution value).
        'postcode_freq_default': float(min(pc_freq.values())),
        'postcode_area_freq_default': float(min(area_freq.values())),
        # Inference defaults for fields a price-less request often omits.
        'floor_count_default': float(df['floor_count'].median()),  # 0 (median/mode; 56% are 0)
        'note': ('Single-row inference must inject postcode_freq/postcode_area_freq from '
                 'these maps (keyed on postcode_district / postcode_area) instead of the '
                 'degenerate 1.0 recompute. floor_count default = training median.'),
    }

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {OUT}")
    print(f"  n_train={N}")
    print(f"  postcode_freq: {len(pc_freq)} districts (SW3={pc_freq.get('SW3',0):.4f}, "
          f"W11={pc_freq.get('W11',0):.4f}); default={stats['postcode_freq_default']:.5f}")
    print(f"  postcode_area_freq: {len(area_freq)} areas (SW={area_freq.get('SW',0):.4f}); "
          f"default={stats['postcode_area_freq_default']:.5f}")
    print(f"  floor_count_default={stats['floor_count_default']}")
    return stats


if __name__ == '__main__':
    main()
