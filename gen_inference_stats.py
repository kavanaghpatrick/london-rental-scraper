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

NOTE: as of the tail-accuracy fix (FIX 1), engineer_features_v20 recovers the REAL
postcode_district from raw `postcode` (no longer the `postcode_normalized.fillna('SW3')`
artifact). These freqs therefore reflect the real district distribution the model is now
fit on (SW3 ≈ 0.08, not 0.88). Inference reproduces these exact maps to match training.

Usage:  python3 gen_inference_stats.py   ->  output/rental_model_canonical_inference.json
"""
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

import retrain_canonical as rc

OUT = Path('output/rental_model_canonical_inference.json')


def _assert_default_shape(stats: dict) -> None:
    """Fail LOUDLY before writing if the freq-default invariant is violated. The reader
    (canonical_predict) and the JS predictor both key off a NUMERIC top-level
    `*_default` that equals min(map). A None default or a nested in-map 'default' is the
    exact shape that crashed np.log1p / broke JS↔Python parity — never write it.
    Shared with retrain_canonical (the SECOND writer) so both stay shape-identical."""
    for base in ('postcode_freq', 'postcode_area_freq'):
        fmap, dflt = stats[base], stats[f'{base}_default']
        assert isinstance(dflt, float), f"{base}_default must be float, got {type(dflt).__name__}"
        assert dflt == float(min(fmap.values())), f"{base}_default must equal min({base}.values())"
        assert 'default' not in fmap, f"{base} map must not embed a nested 'default' key"


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
                 'degenerate recompute. floor_count default = training median.'),
    }

    # FIX 2 (GATED): neighborhood `area` maps — only when the area features are selected.
    #   area_freq_map        : count(area_key)/N  (mirrors engineer_features_v20.area_freq)
    #   area_target_encoding : full-train smoothed log-price map (the INFERENCE map;
    #     training rows use the OOF encoding — see compute_area_target_encoding_oof).
    if rc.v20.AREA_ENABLED:
        area_key_counts = df['area_key'].value_counts()
        area_freq_map = {str(k): v / N for k, v in area_key_counts.items()}
        area_te_map, area_te_global = rc.v20.build_area_target_map(df['area_key'].values, df['price_pcm'].values)
        stats['area_freq'] = area_freq_map
        stats['area_freq_default'] = float(min(area_freq_map.values()))
        stats['area_target_encoding'] = area_te_map
        stats['area_te_default'] = float(area_te_global)

    _assert_default_shape(stats)  # invariant: numeric sibling defaults, no nested 'default'

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {OUT}")
    print(f"  n_train={N}")
    print(f"  postcode_freq: {len(pc_freq)} districts (SW3={pc_freq.get('SW3',0):.4f}, "
          f"W11={pc_freq.get('W11',0):.4f}); default={stats['postcode_freq_default']:.5f}")
    print(f"  postcode_area_freq: {len(area_freq)} areas (SW={area_freq.get('SW',0):.4f}); "
          f"default={stats['postcode_area_freq_default']:.5f}")
    if rc.v20.AREA_ENABLED:
        print(f"  area_freq: {len(stats['area_freq'])} neighborhoods; default={stats['area_freq_default']:.5f}")
        print(f"  area_target_encoding: {len(stats['area_target_encoding'])} neighborhoods; "
              f"global_prior(log)={stats['area_te_default']:.4f}")
    print(f"  floor_count_default={stats['floor_count_default']}")
    return stats


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Generate canonical inference freq-map stats.')
    ap.add_argument('--db', default='output/rentals.db',
                    help='SQLite DB to compute training freqs from (must match the retrain DB).')
    args = ap.parse_args()
    main(db=args.db)
