"""
Retrain the CANONICAL rental price model (v20, chosen by bake-off) on the full
canonical dataset and save to a DETERMINISTIC path.

Canonical artifacts (the EXACT filenames artifacts(#7) + automation(#9) must load):
    output/rental_model_canonical.pkl           <- pickled XGBRegressor
    output/rental_model_canonical_features.pkl  <- list[str] feature order
    output/rental_model_canonical_meta.json     <- version, metrics, n, sha-ish info

The canonical model IS v20 (rental_price_models_v20.py feature engineering).
We reuse v20's own load/engineer/feature functions so the artifact is byte-for-byte
consistent with the production training code.

Data: per DATA_LAYER_CONTRACT.md §5, train on the recency-independent set (NO
is_active filter) from a COPY of output/rentals.db.

Usage:
    python3 retrain_canonical.py --db output/rentals_modeler_copy.db
"""
import argparse
import json
import pickle
import time
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

import rental_price_models_v20 as v20

CANON_VERSION = 'v20'
OUT = Path('output')
MODEL_PATH = OUT / 'rental_model_canonical.pkl'
FEATURES_PATH = OUT / 'rental_model_canonical_features.pkl'
META_PATH = OUT / 'rental_model_canonical_meta.json'

# Shared quality-filter constants (DATA_LAYER_CONTRACT §5: NO is_active filter)
PPSF_MIN, PPSF_MAX = 3, 30
PRICE_MIN_PCM = 500
SQFT_MIN, SQFT_MAX = 150, 10000
SQFT_PER_BED_MIN = 70


def load_recency_independent(db_path):
    """Load + clean + dedupe using v20's pipeline but WITHOUT is_active (frozen snapshot)."""
    query = """
        SELECT
            bedrooms, bathrooms, size_sqft,
            postcode, area, property_type, address,
            price_pcm, latitude, longitude, features, description, source,
            property_type_std, let_type, postcode_normalized,
            postcode_inferred, agent_brand,
            floor_count, has_roof_terrace, has_basement, has_ground,
            has_first_floor, has_second_floor, has_third_floor, has_fourth_plus
        FROM listings
        WHERE size_sqft > 0 AND bedrooms IS NOT NULL AND price_pcm > 0
        AND (is_short_let = 0 OR is_short_let IS NULL)
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"Raw recency-independent rows: {len(df)}")

    df['ppsf'] = df['price_pcm'] / df['size_sqft']
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    mask = (
        (sqft_per_bed < SQFT_PER_BED_MIN) | (df['price_pcm'] > 100000) |
        (df['size_sqft'] < SQFT_MIN) | (df['size_sqft'] > SQFT_MAX) |
        (df['ppsf'] < PPSF_MIN) | (df['ppsf'] > PPSF_MAX) |
        (df['price_pcm'] < PRICE_MIN_PCM)
    )
    df = df[~mask].copy()
    print(f"After quality filter: {len(df)}")

    # v20 also excludes short-lets/serviced via is_short_let_or_serviced (ppsf>20 etc.)
    short = df.apply(v20.is_short_let_or_serviced, axis=1)
    df = df[~short].copy()
    print(f"After short-let/serviced exclusion: {len(df)}")

    # Cross-source dedupe (identical to v20)
    df['postcode_district'] = df['postcode'].str.extract(r'^([A-Z]+\d+[A-Z]?)', expand=False)
    source_priority = {'savills': 0, 'knightfrank': 1, 'chestertons': 2, 'foxtons': 3, 'rightmove': 4}
    df['source_rank'] = df['source'].map(source_priority).fillna(5)
    df = df.sort_values('source_rank')
    df = df.drop_duplicates(subset=['price_pcm', 'size_sqft', 'bedrooms', 'postcode_district'], keep='first')
    df = df.drop(columns=['source_rank']).reset_index(drop=True)
    print(f"After dedupe: {len(df)}\n")
    return df


def cv_metrics(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    yl = np.log1p(y)
    preds, actual = [], []
    for tr, va in kf.split(X):
        m = v20.build_xgboost()
        m.fit(X.iloc[tr], yl.iloc[tr])
        preds.extend(np.expm1(m.predict(X.iloc[va])))
        actual.extend(y.iloc[va].values)
    preds, actual = np.array(preds), np.array(actual)
    return {
        'R2': float(r2_score(actual, preds)),
        'RMSE': float(np.sqrt(mean_squared_error(actual, preds))),
        'MAE': float(mean_absolute_error(actual, preds)),
        'MAPE': float(np.mean(np.abs((actual - preds) / actual)) * 100),
        'Median_APE': float(np.median(np.abs((actual - preds) / actual)) * 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='output/rentals_modeler_copy.db')
    args = ap.parse_args()

    print("=" * 70)
    print(f"RETRAIN CANONICAL MODEL = {CANON_VERSION}")
    print("=" * 70)

    df = load_recency_independent(args.db)
    df = v20.engineer_features_v20(df)
    feat_cols = v20.get_feature_columns_v20(df)

    X = df[feat_cols].copy()
    for c in X.columns:
        if X[c].dtype == 'bool':
            X[c] = X[c].astype(int)
        elif X[c].dtype == 'object':
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)
    y = df['price_pcm']

    print(f"Features: {len(feat_cols)}   Samples: {len(X)}\n")

    print("[5-fold CV on full canonical data]")
    metrics = cv_metrics(X, y)
    print(f"  R2={metrics['R2']:.4f}  RMSE=£{metrics['RMSE']:,.0f}  MAE=£{metrics['MAE']:,.0f}  "
          f"MAPE={metrics['MAPE']:.1f}%  MedAPE={metrics['Median_APE']:.1f}%\n")

    print("[Fitting final model on ALL rows]")
    t0 = time.time()
    final = v20.build_xgboost()
    final.fit(X, np.log1p(y))
    fit_time = round(time.time() - t0, 1)
    print(f"  fit in {fit_time}s")

    OUT.mkdir(exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final, f)
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(feat_cols, f)

    meta = {
        'canonical_version': CANON_VERSION,
        'trained_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'db_source': 'output/rentals.db (frozen snapshot, last scrape 2026-01-16)',
        'train_filter': 'recency-independent (no is_active) per DATA_LAYER_CONTRACT.md §5',
        'n_samples': int(len(X)),
        'n_features': len(feat_cols),
        'target': 'log1p(price_pcm), inverse expm1 at predict',
        'cv_metrics_5fold': metrics,
        'xgb_params': final.get_params(),
        'artifact_paths': {
            'model': str(MODEL_PATH),
            'features': str(FEATURES_PATH),
        },
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    # ── Emit the inference-stats sidecar IN THE SAME RUN ──────────────────────
    # canonical_predict.build_features(inference=True) needs the training frequency
    # maps (output/rental_model_canonical_inference.json) to fix the single-row
    # postcode_freq/area_freq degeneration (task #30). Computing it here — from the
    # SAME engineered `df` the model was just fit on — guarantees the freq maps can
    # NEVER drift from the pkl. (A separate post-retrain step risks a fresh pkl +
    # stale inference map → the £3,430 prime-market bug returns silently.)
    INFERENCE_PATH = OUT / 'rental_model_canonical_inference.json'
    pc_freq = {str(k): v / len(df) for k, v in df['postcode_district'].value_counts().items()}
    area_freq = {str(k): v / len(df) for k, v in df['postcode_area'].value_counts().items()}
    inference_stats = {
        'canonical_version': CANON_VERSION,
        'n_train': int(len(df)),
        'postcode_freq': pc_freq,
        'postcode_area_freq': area_freq,
        'postcode_freq_default': float(min(pc_freq.values())),
        'postcode_area_freq_default': float(min(area_freq.values())),
        'floor_count_default': float(df['floor_count'].median()),
        'note': ('Single-row inference injects postcode_freq/postcode_area_freq from these '
                 'training maps (keyed on postcode_district / postcode_area) instead of the '
                 'degenerate 1.0 recompute. Regenerated atomically with the pkl every retrain.'),
    }
    with open(INFERENCE_PATH, 'w') as f:
        json.dump(inference_stats, f, indent=2)

    print(f"\nSaved canonical artifacts:")
    print(f"  {MODEL_PATH}")
    print(f"  {FEATURES_PATH}")
    print(f"  {META_PATH}")
    print(f"  {INFERENCE_PATH}  (inference freq maps — regenerated with the pkl)")
    return meta


if __name__ == '__main__':
    main()
