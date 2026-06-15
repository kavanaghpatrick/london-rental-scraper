"""
Model Bake-Off: v15 vs v16 vs v19 vs v20 on ONE common holdout.

Task #6 (modeler). Decides the CANONICAL rental price model version.

DESIGN
------
The four candidate scripts each have their OWN data-cleaning + feature-engineering
pipeline. To compare them fairly we must hold TWO things constant:

  1. The exact set of test rows (the holdout).
  2. The train/test boundary (which rows a model may learn from).

We therefore:
  - Load the raw listings ONCE from output/rentals.db (canonical DB).
  - Apply the COMMON quality filter + cross-source dedupe that ALL four versions
    share (identical thresholds in every script). We do NOT apply v16/v20's
    short-let/serviced exclusion here, because v15/v19 don't, and excluding rows
    would change the holdout per-version. Instead each model's own engineer_*()
    runs on the shared rows.
  - Make ONE 80/20 split with a fixed seed -> shared train_idx / test_idx.
  - For each version: run that version's feature engineering on the FULL shared
    frame (engineering is row-independent except for a few group-frequency/qcut
    encodings which we accept as-is since they're computed the same way each
    version does internally), select the version's feature columns, then fit on
    train rows and score on the identical test rows.
  - Report RMSE / MAE / R^2 / MAPE / Median-APE / feature-count / train-time.

All four use the SAME XGBoost hyper-params (build_xgboost defaults shared across
scripts) and the SAME log1p(price) target, so the ONLY thing that varies is the
feature set + that version's row filtering quirks. That is exactly the
comparison the decision needs.

USAGE
-----
    python3 bakeoff_v15_v16_v19_v20.py                 # full bake-off
    python3 bakeoff_v15_v16_v19_v20.py --db PATH       # override DB path
    python3 bakeoff_v15_v16_v19_v20.py --json out.json # write metrics JSON
"""

import argparse
import json
import time
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# Import the four candidate modules (they share function names, so import as ns).
# v16 + v19 are bake-off LOSERS. Model-version archiving is on hold pending the lead's
# final user confirmation, so they may sit at repo root OR under the cleanup archive at
# any given moment. We append the archive dir to sys.path as a fallback: root copies win
# when present, the archive copies resolve the imports otherwise. Harmless no-op once the
# losers settle at root. This keeps the frozen decision-record harness runnable either way.
import sys as _sys
from pathlib import Path as _Path
_ARCHIVE = _Path(__file__).resolve().parent / 'archive' / '_cleanup_2026-06-15' / 'old_model_scripts'
if _ARCHIVE.is_dir() and str(_ARCHIVE) not in _sys.path:
    _sys.path.append(str(_ARCHIVE))   # appended (not prepended): root copies take priority

import rental_price_models_v15 as v15
import rental_price_models_v16 as v16          # bake-off loser (root or archive)
import train_v19_extend_v18 as v19             # bake-off loser (root or archive)
import rental_price_models_v20 as v20

# Shared quality-filter constants (identical across all four scripts)
PPSF_MIN, PPSF_MAX = 3, 30
PRICE_MIN_PCM = 500
SQFT_MIN, SQFT_MAX = 150, 10000
SQFT_PER_BED_MIN = 70

RANDOM_STATE = 42
TEST_SIZE = 0.20


def shared_xgb():
    """The exact XGBoost config every candidate script uses for its main model."""
    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective='reg:absoluteerror',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def load_common_frame(db_path):
    """Load + clean + dedupe ONCE using the filter all four versions share.

    Returns a DataFrame with every column any version's engineer_*() needs.
    """
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
    # DATA_LAYER_CONTRACT.md §5: FROZEN snapshot (last scrape 2026-01-16). is_active=1
    # is a serving/recency gate, NOT a training filter — it drops ~1,600 valid rows
    # for no modeling benefit. dataeng recommends the recency-independent set (6,402).
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"Raw trainable rows: {len(df)}")

    # Common quality filter (identical thresholds in v15/v16/v19/v20)
    df['ppsf'] = df['price_pcm'] / df['size_sqft']
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    mask = (
        (sqft_per_bed < SQFT_PER_BED_MIN) |
        (df['price_pcm'] > 100000) |
        (df['size_sqft'] < SQFT_MIN) |
        (df['size_sqft'] > SQFT_MAX) |
        (df['ppsf'] < PPSF_MIN) |
        (df['ppsf'] > PPSF_MAX) |
        (df['price_pcm'] < PRICE_MIN_PCM)
    )
    df = df[~mask].copy()
    print(f"After common quality filter: {len(df)}")

    # Common cross-source dedupe (identical in all four)
    df['postcode_district'] = df['postcode'].str.extract(r'^([A-Z]+\d+[A-Z]?)', expand=False)
    source_priority = {'savills': 0, 'knightfrank': 1, 'chestertons': 2, 'foxtons': 3, 'rightmove': 4}
    df['source_rank'] = df['source'].map(source_priority).fillna(5)
    df = df.sort_values('source_rank')
    df = df.drop_duplicates(subset=['price_pcm', 'size_sqft', 'bedrooms', 'postcode_district'], keep='first')
    df = df.drop(columns=['source_rank'])
    df = df.reset_index(drop=True)
    print(f"After common dedupe: {len(df)}  <- shared holdout universe\n")
    return df


def metrics(actual, preds):
    actual = np.asarray(actual, dtype=float)
    preds = np.asarray(preds, dtype=float)
    return {
        'RMSE': float(np.sqrt(mean_squared_error(actual, preds))),
        'MAE': float(mean_absolute_error(actual, preds)),
        'R2': float(r2_score(actual, preds)),
        'MAPE': float(np.mean(np.abs((actual - preds) / actual)) * 100),
        'Median_APE': float(np.median(np.abs((actual - preds) / actual)) * 100),
    }


def run_version(name, engineer_fn, feature_fn, base_df, train_idx, test_idx):
    """Engineer features for ONE version, fit on train rows, score on test rows."""
    t0 = time.time()
    df = engineer_fn(base_df.copy())

    feat_cols = feature_fn(df)
    # ensure any V18/V19 declared-but-missing columns are zero-filled
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0

    X = df[feat_cols].copy()
    for c in X.columns:
        if X[c].dtype == 'bool':
            X[c] = X[c].astype(int)
        elif X[c].dtype == 'object':
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)
    y = df['price_pcm']

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = shared_xgb()
    model.fit(X_train, np.log1p(y_train))
    preds = np.expm1(model.predict(X_test))
    train_time = time.time() - t0

    m = metrics(y_test.values, preds)
    m['n_features'] = len(feat_cols)
    m['train_time_s'] = round(train_time, 1)
    m['n_train'] = len(X_train)
    m['n_test'] = len(X_test)
    print(f"[{name}] feats={m['n_features']:>3}  RMSE=£{m['RMSE']:,.0f}  "
          f"MAE=£{m['MAE']:,.0f}  R2={m['R2']:.4f}  MAPE={m['MAPE']:.1f}%  "
          f"MedAPE={m['Median_APE']:.1f}%  time={m['train_time_s']}s")
    return m


def v19_features_for(df):
    """v19's feature list = V18_FEATURES (live features.json) + 3 mews features."""
    return v19.get_v19_features()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='output/rentals.db')
    ap.add_argument('--json', default='output/bakeoff_results.json')
    args = ap.parse_args()

    print("=" * 78)
    print("MODEL BAKE-OFF: v15 vs v16 vs v19 vs v20  (ONE common holdout)")
    print("=" * 78)
    print(f"DB: {args.db}  seed={RANDOM_STATE}  test_size={TEST_SIZE}\n")

    base_df = load_common_frame(args.db)

    # ONE shared split, reused by every version
    idx = np.arange(len(base_df))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Shared split: {len(train_idx)} train / {len(test_idx)} test\n")
    print("-" * 78)

    results = {}
    results['v15'] = run_version('v15', v15.engineer_features_v15, v15.get_feature_columns,
                                 base_df, train_idx, test_idx)
    results['v16'] = run_version('v16', v16.engineer_features_v16, v16.get_feature_columns_v16,
                                 base_df, train_idx, test_idx)
    # v19.engineer_features returns (df, mews_info) -> unwrap to df
    results['v19'] = run_version('v19', lambda d: v19.engineer_features(d)[0],
                                 v19_features_for, base_df, train_idx, test_idx)
    results['v20'] = run_version('v20', v20.engineer_features_v20, v20.get_feature_columns_v20,
                                 base_df, train_idx, test_idx)

    print("-" * 78)
    print("\nSUMMARY (lower RMSE/MAE/MAPE better, higher R2 better)\n")
    hdr = f"{'ver':<5}{'feats':>6}{'RMSE':>12}{'MAE':>10}{'R2':>9}{'MAPE':>8}{'MedAPE':>9}{'time':>8}"
    print(hdr)
    print("-" * len(hdr))
    for v, m in results.items():
        print(f"{v:<5}{m['n_features']:>6}{('£%0.0f' % m['RMSE']):>12}{('£%0.0f' % m['MAE']):>10}"
              f"{m['R2']:>9.4f}{('%0.1f%%' % m['MAPE']):>8}{('%0.1f%%' % m['Median_APE']):>9}"
              f"{('%0.1fs' % m['train_time_s']):>8}")

    best_r2 = max(results, key=lambda v: results[v]['R2'])
    best_rmse = min(results, key=lambda v: results[v]['RMSE'])
    print(f"\nBest R2:   {best_r2}")
    print(f"Best RMSE: {best_rmse}")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, 'w') as f:
        json.dump({'config': {'db': args.db, 'seed': RANDOM_STATE, 'test_size': TEST_SIZE,
                              'n_total': len(base_df), 'n_train': len(train_idx), 'n_test': len(test_idx)},
                   'results': results}, f, indent=2)
    print(f"\nWrote {args.json}")
    return results


if __name__ == '__main__':
    main()
