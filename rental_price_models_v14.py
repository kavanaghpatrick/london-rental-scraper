"""
London Rental Price Prediction Models V14 - Streamlined Feature Set

A cleaner, more interpretable model with 47 focused features:
- Core: size_sqft, bedrooms, bathrooms, floor_count
- Floor flags: has_basement, has_lower_ground, has_ground, etc.
- Amenities: has_garden, has_porter, has_gym, has_pool, etc.
- Property type: is_period, is_penthouse, is_lateral, has_ensuite
- Location keywords: is_square, is_place, is_crescent, is_gate, near_hyde_park
- Neighborhood: is_notting_hill, is_belgravia, is_knightsbridge, is_kensington, is_mayfair, is_chelsea
- Engineered: log_size, sqft_per_bed, luxury_score, location_score
- Target encoding: postcode_ppsf (blended), source_ppsf, is_prime

Saves:
- rental_model_v14.pkl: Trained XGBoost model
- rental_model_v14_info.pkl: PPSF encodings, prime districts, params
- rental_model_v14_features.pkl: Feature names list

Usage:
    python rental_price_models_v14.py
    python rental_price_models_v14.py --quick    # Skip Optuna tuning
"""

import pandas as pd
import numpy as np
import sqlite3
import pickle
import argparse
import re
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed, skipping hyperparameter tuning")

OUTPUT_DIR = Path('output')

# Prime London districts (high-value areas)
PRIME_DISTRICTS = ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW1', 'NW3', 'NW8']


def load_data():
    """Load rental data from SQLite."""
    conn = sqlite3.connect('output/rentals.db')
    df = pd.read_sql("""
        SELECT
            id, source, address, postcode, price_pcm, size_sqft, bedrooms, bathrooms,
            floor_count, has_roof_terrace, has_basement, has_lower_ground, has_ground,
            has_mezzanine, has_first_floor, has_second_floor, has_third_floor, has_fourth_plus,
            features, description
        FROM listings
        WHERE size_sqft > 0 AND size_sqft IS NOT NULL
          AND bedrooms IS NOT NULL
          AND price_pcm > 0 AND price_pcm IS NOT NULL
          AND postcode IS NOT NULL AND postcode != ''
    """, conn)
    conn.close()

    print(f"Loaded {len(df)} records with size, bedrooms, price, and postcode")
    return df


def extract_postcode_district(postcode):
    """Extract district from postcode (e.g., SW3 from SW3 4TX)."""
    if not postcode or pd.isna(postcode):
        return None
    match = re.match(r'^([A-Z]{1,2}\d{1,2}[A-Z]?)', str(postcode).upper().strip())
    return match.group(1) if match else None


def extract_postcode_area(postcode):
    """Extract area from postcode (e.g., SW from SW3)."""
    if not postcode or pd.isna(postcode):
        return None
    match = re.match(r'^([A-Z]{1,2})', str(postcode).upper().strip())
    return match.group(1) if match else None


def parse_amenities(features_str, description_str=''):
    """Extract amenity flags from features and description text."""
    text = ''
    if features_str and not pd.isna(features_str):
        text += str(features_str).lower() + ' '
    if description_str and not pd.isna(description_str):
        text += str(description_str).lower()

    return {
        'has_garden': int('garden' in text),
        'has_terrace': int('terrace' in text and 'roof terrace' not in text),
        'has_balcony': int('balcony' in text),
        'has_parking': int('parking' in text or 'garage' in text),
        'has_porter': int('porter' in text or 'concierge' in text or '24 hour' in text or '24-hour' in text),
        'has_concierge': int('concierge' in text),
        'has_gym': int('gym' in text or 'fitness' in text),
        'has_pool': int('pool' in text or 'swimming' in text),
        'has_lift': int('lift' in text or 'elevator' in text),
        'has_cinema': int('cinema' in text or 'home theatre' in text or 'screening' in text),
        'has_ensuite': int('ensuite' in text or 'en-suite' in text or 'en suite' in text),
        'is_period': int('period' in text or 'victorian' in text or 'georgian' in text or 'edwardian' in text),
        'is_penthouse': int('penthouse' in text),
        'is_lateral': int('lateral' in text),
    }


def parse_address_features(address):
    """Extract location keywords from address."""
    if not address or pd.isna(address):
        return {k: 0 for k in ['is_square', 'is_place', 'is_crescent', 'is_gate', 'near_hyde_park',
                               'is_notting_hill', 'is_belgravia', 'is_knightsbridge',
                               'is_kensington', 'is_mayfair', 'is_chelsea']}

    addr = str(address).lower()
    return {
        'is_square': int(' square' in addr or 'sq.' in addr),
        'is_place': int(' place' in addr),
        'is_crescent': int('crescent' in addr),
        'is_gate': int(' gate' in addr or ' gates' in addr),
        'near_hyde_park': int('hyde park' in addr or 'park lane' in addr or 'marble arch' in addr),
        'is_notting_hill': int('notting hill' in addr or 'portobello' in addr),
        'is_belgravia': int('belgravia' in addr or 'eaton' in addr or 'chester' in addr and 'square' in addr),
        'is_knightsbridge': int('knightsbridge' in addr),
        'is_kensington': int('kensington' in addr),
        'is_mayfair': int('mayfair' in addr or 'grosvenor' in addr or 'berkeley' in addr),
        'is_chelsea': int('chelsea' in addr or 'sloane' in addr),
    }


def engineer_features(df):
    """Engineer V14 features."""
    print("\n[FEATURE ENGINEERING V14]")

    # Extract postcode components
    df['postcode_district'] = df['postcode'].apply(extract_postcode_district)
    df['postcode_area'] = df['postcode'].apply(extract_postcode_area)

    # Filter rows with valid postcode district
    valid_postcode = df['postcode_district'].notna()
    print(f"  Valid postcodes: {valid_postcode.sum()} / {len(df)}")
    df = df[valid_postcode].copy()

    # Quality filters
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    ppsf = df['price_pcm'] / df['size_sqft']

    # Remove outliers
    mask_quality = (
        (sqft_per_bed >= 70) &  # At least 70 sqft per bedroom
        (ppsf >= 1.0) & (ppsf <= 25.0) &  # Reasonable price per sqft
        (df['price_pcm'] <= 60000) &  # Max £60k/month
        (df['size_sqft'] >= 150)  # Min 150 sqft
    )
    df = df[mask_quality].copy()
    print(f"  After quality filters: {len(df)} records")

    # Fill floor features
    floor_cols = ['floor_count', 'has_roof_terrace', 'has_basement', 'has_lower_ground',
                  'has_ground', 'has_mezzanine', 'has_first_floor', 'has_second_floor',
                  'has_third_floor', 'has_fourth_plus']
    for col in floor_cols:
        df[col] = df[col].fillna(0).astype(int)

    # Parse amenities
    amenity_dicts = df.apply(lambda row: parse_amenities(row['features'], row.get('description', '')), axis=1)
    amenity_df = pd.DataFrame(amenity_dicts.tolist())
    for col in amenity_df.columns:
        df[col] = amenity_df[col].values

    # Parse address features
    addr_dicts = df['address'].apply(parse_address_features)
    addr_df = pd.DataFrame(addr_dicts.tolist())
    for col in addr_df.columns:
        df[col] = addr_df[col].values

    # Engineered features
    df['log_size'] = np.log1p(df['size_sqft'])
    df['sqft_per_bed'] = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)

    # Luxury score (count of luxury amenities)
    luxury_amenities = ['has_pool', 'has_cinema', 'has_gym', 'has_porter', 'has_concierge']
    df['luxury_score'] = df[luxury_amenities].sum(axis=1)

    # Location score (count of premium location indicators)
    location_indicators = ['is_knightsbridge', 'is_mayfair', 'is_belgravia', 'is_chelsea',
                          'is_kensington', 'is_notting_hill', 'near_hyde_park', 'is_square']
    df['location_score'] = df[location_indicators].sum(axis=1)

    # Is prime district
    df['is_prime'] = df['postcode_district'].apply(lambda x: 1 if x in PRIME_DISTRICTS else 0)

    print(f"  Prime district listings: {df['is_prime'].sum()} ({df['is_prime'].mean()*100:.1f}%)")

    return df


def compute_ppsf_encodings(df):
    """Compute price-per-sqft encodings at district, area, and source levels."""
    df = df.copy()
    df['ppsf'] = df['price_pcm'] / df['size_sqft']

    # Global median
    global_ppsf_median = df['ppsf'].median()

    # District-level PPSF
    district_stats = df.groupby('postcode_district').agg({
        'ppsf': 'median',
        'id': 'count'
    }).rename(columns={'ppsf': 'district_ppsf', 'id': 'district_count'})

    # Area-level PPSF
    area_stats = df.groupby('postcode_area')['ppsf'].median()

    # Source-level PPSF
    source_stats = df.groupby('source')['ppsf'].median()

    # Blended district PPSF (blend with area when few samples)
    district_ppsf = {}
    district_count = {}
    for district in df['postcode_district'].unique():
        d_data = district_stats.loc[district] if district in district_stats.index else None
        if d_data is not None:
            d_ppsf = d_data['district_ppsf']
            d_count = d_data['district_count']

            area = extract_postcode_area(district)
            a_ppsf = area_stats.get(area, global_ppsf_median)

            # Blend: more weight to district when more samples
            weight = min(d_count / 10, 1.0)
            blended_ppsf = weight * d_ppsf + (1 - weight) * a_ppsf

            district_ppsf[district] = blended_ppsf
            district_count[district] = d_count

    return {
        'district_ppsf': district_ppsf,
        'district_count': district_count,
        'area_ppsf': area_stats.to_dict(),
        'source_ppsf': source_stats.to_dict(),
        'global_ppsf_median': global_ppsf_median,
    }


def apply_ppsf_encodings(df, ppsf_info):
    """Apply PPSF encodings to dataframe."""
    df = df.copy()

    # Postcode PPSF
    df['postcode_ppsf'] = df['postcode_district'].map(ppsf_info['district_ppsf'])
    df['postcode_ppsf'] = df['postcode_ppsf'].fillna(ppsf_info['global_ppsf_median'])

    # Source PPSF
    df['source_ppsf'] = df['source'].map(ppsf_info['source_ppsf'])
    df['source_ppsf'] = df['source_ppsf'].fillna(ppsf_info['global_ppsf_median'])

    # Interaction features
    df['size_x_ppsf'] = df['log_size'] * df['postcode_ppsf']
    df['size_x_prime'] = df['size_sqft'] * df['is_prime']

    return df


def get_feature_columns():
    """Get V14 feature column names in order.

    NOTE: Removed leaky features that use price-derived encodings:
    - postcode_ppsf (computed from target)
    - source_ppsf (computed from target)
    - size_x_ppsf (uses postcode_ppsf)
    """
    return [
        # Core
        'size_sqft', 'bedrooms', 'bathrooms', 'floor_count',
        # Floor flags
        'has_basement', 'has_lower_ground', 'has_ground', 'has_mezzanine',
        'has_first_floor', 'has_second_floor', 'has_third_floor', 'has_fourth_plus',
        'has_roof_terrace',
        # Amenities
        'has_garden', 'has_terrace', 'has_balcony', 'has_parking',
        'has_porter', 'has_concierge', 'has_gym', 'has_pool', 'has_lift',
        # Property type
        'is_period', 'is_penthouse', 'has_ensuite',
        # Address features
        'is_square', 'is_lateral', 'near_hyde_park', 'is_place', 'has_cinema',
        'is_crescent', 'is_gate',
        # Neighborhood
        'is_notting_hill', 'is_belgravia', 'is_knightsbridge', 'is_kensington',
        'is_mayfair', 'is_chelsea',
        # Engineered (no leakage)
        'log_size', 'sqft_per_bed', 'is_prime',
        # Interactions (no leakage)
        'size_x_prime',
        # Scores
        'luxury_score', 'location_score',
    ]


def evaluate_model(model, X, y, cv=5):
    """Evaluate with cross-validation."""
    y_log = np.log1p(y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    metrics = {'r2': [], 'mae': [], 'mape': [], 'median_ape': []}

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.clip(y_pred, 500, 100000)

        metrics['r2'].append(r2_score(y_val, y_pred))
        metrics['mae'].append(mean_absolute_error(y_val, y_pred))
        ape = np.abs((y_val - y_pred) / y_val) * 100
        metrics['mape'].append(ape.mean())
        metrics['median_ape'].append(np.median(ape))

    return {k: np.mean(v) for k, v in metrics.items()}


def train_model(df, quick=False):
    """Train XGBoost V14 model."""
    print("\n" + "="*70)
    print("TRAINING XGBOOST V14")
    print("="*70)

    feature_cols = get_feature_columns()

    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            print(f"  WARNING: Missing feature {col}, filling with 0")
            df[col] = 0

    X = df[feature_cols].copy()
    y = df['price_pcm'].copy()

    # Fill any remaining NaN
    X = X.fillna(0)

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")

    # Default params (from v14_info.pkl)
    default_params = {
        'n_estimators': 231,
        'max_depth': 5,
        'learning_rate': 0.034,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 2,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }

    if not quick and HAS_OPTUNA:
        print("\n[OPTUNA TUNING]")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 400),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
            }
            model = XGBRegressor(**params)
            metrics = evaluate_model(model, X, y, cv=3)
            return metrics['r2']

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=True)
        best_params = {**default_params, **study.best_params}
        print(f"  Best R²: {study.best_value:.4f}")
    else:
        best_params = default_params
        print("\n[USING DEFAULT PARAMS]")

    # Train final model
    model = XGBRegressor(**best_params)
    metrics = evaluate_model(model, X, y, cv=5)

    print(f"\n[V14 RESULTS]")
    print(f"  R²:         {metrics['r2']:.4f}")
    print(f"  MAE:        £{metrics['mae']:,.0f}")
    print(f"  MAPE:       {metrics['mape']:.1f}%")
    print(f"  Median APE: {metrics['median_ape']:.1f}%")

    # Train on full data
    model.fit(X, np.log1p(y))

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n[TOP 20 FEATURES]")
    for _, row in importance.head(20).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")

    return model, metrics, best_params, importance


def save_model(model, ppsf_info, best_params, feature_names, train_size):
    """Save model and metadata."""
    # Save model
    with open(OUTPUT_DIR / 'rental_model_v14.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save info
    info = {
        **ppsf_info,
        'prime_districts': PRIME_DISTRICTS,
        'feature_names': feature_names,
        'log_target': True,
        'best_params': best_params,
        'trained_on': 'deduplicated_data',
        'train_size': train_size,
    }
    with open(OUTPUT_DIR / 'rental_model_v14_info.pkl', 'wb') as f:
        pickle.dump(info, f)

    # Save feature names
    with open(OUTPUT_DIR / 'rental_model_v14_features.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    print(f"\nSaved model to {OUTPUT_DIR / 'rental_model_v14.pkl'}")
    print(f"Saved info to {OUTPUT_DIR / 'rental_model_v14_info.pkl'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Skip Optuna tuning')
    args = parser.parse_args()

    # Load data
    df = load_data()

    # Engineer features
    df = engineer_features(df)

    # Compute PPSF encodings
    ppsf_info = compute_ppsf_encodings(df)

    # Apply encodings
    df = apply_ppsf_encodings(df, ppsf_info)

    print(f"\nFinal dataset: {len(df)} records")
    print(f"Postcode districts: {df['postcode_district'].nunique()}")
    print(f"Sources: {df['source'].unique().tolist()}")

    # Train model
    model, metrics, best_params, importance = train_model(df, quick=args.quick)

    # Save
    save_model(model, ppsf_info, best_params, get_feature_columns(), len(df))

    # Version history
    print("\n" + "="*70)
    print("VERSION HISTORY")
    print("="*70)
    print("V1 (baseline):    R²=0.658, MAE=£2,589, MAPE=28.2%")
    print("V7 (Optuna):      R²=0.834, MAE=£1,584, MAPE=20.3%")
    print("V10 (Floor):      R²=0.839, MAE=£1,560, MAPE=19.6%")
    print(f"V14 (Streamlined): R²={metrics['r2']:.3f}, MAE=£{metrics['mae']:,.0f}, MAPE={metrics['mape']:.1f}%")


if __name__ == '__main__':
    main()
