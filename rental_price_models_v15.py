"""
London Rental Price Prediction Models V15 - NO LEAKAGE VERSION

This model removes ALL target-derived features that caused leakage in V6-V14:
- NO is_luxury (derived from price_pcm > 10000)
- NO luxury_*_interaction features
- NO global target encoding (uses K-fold CV encoding instead)
- NO ppsf_encoded features (ppsf = price/sqft contains target)
- NO features that use encoded values with target leakage

Features retained (no leakage):
- Core: bedrooms, bathrooms, size_sqft
- Location: postcode one-hot, lat/lon distances
- Amenities: from features JSON (no price info)
- Agent: brand name (categorical)
- Property type: from property_type_std
- Floor data: floor_count, floor flags

Usage:
    python rental_price_models_v15.py
    python rental_price_models_v15.py --quick
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import argparse
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

OUTPUT_DIR = Path('output')

# Location data (no price info - just coordinates)
TUBE_STATIONS = {
    'South Kensington': (51.4941, -0.1738),
    'Sloane Square': (51.4924, -0.1565),
    'Knightsbridge': (51.5015, -0.1607),
    'Hyde Park Corner': (51.5027, -0.1527),
    'Green Park': (51.5067, -0.1428),
    'Bond Street': (51.5142, -0.1494),
    'Notting Hill Gate': (51.5094, -0.1967),
    'High Street Kensington': (51.5009, -0.1925),
    'Earls Court': (51.4914, -0.1934),
    'Gloucester Road': (51.4945, -0.1829),
    'St Johns Wood': (51.5347, -0.1740),
    'Hampstead': (51.5566, -0.1780),
    'Baker Street': (51.5226, -0.1571),
    'Victoria': (51.4965, -0.1447),
    'Westminster': (51.5014, -0.1248),
    'Paddington': (51.5154, -0.1755),
}

POSTCODE_CENTROIDS = {
    'SW1': (51.4970, -0.1400), 'SW3': (51.4900, -0.1680), 'SW5': (51.4920, -0.1940),
    'SW7': (51.4950, -0.1780), 'SW10': (51.4830, -0.1820), 'SW11': (51.4650, -0.1650),
    'W1': (51.5150, -0.1450), 'W2': (51.5150, -0.1780), 'W8': (51.5010, -0.1920),
    'W11': (51.5150, -0.2050), 'W14': (51.4950, -0.2100),
    'NW1': (51.5350, -0.1550), 'NW3': (51.5550, -0.1780), 'NW8': (51.5330, -0.1750),
    'EC1': (51.5230, -0.1020), 'EC2': (51.5180, -0.0830),
    'WC1': (51.5230, -0.1200), 'WC2': (51.5110, -0.1220),
    'E1': (51.5150, -0.0720), 'E14': (51.5070, -0.0200), 'SE1': (51.5010, -0.1060),
}

CITY_CENTER = (51.5074, -0.1278)

# Premium postcodes (known from domain knowledge, not from price data)
PRIME_POSTCODES = ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW3', 'NW8']

# Premium agents (known brand reputation, not from price data)
PREMIUM_AGENTS = ['Knight Frank', 'Savills', 'Harrods Estates', 'Sotheby',
                  'Beauchamp Estates', 'Strutt & Parker', 'Chestertons']

AMENITY_FEATURES = [
    'has_balcony', 'has_terrace', 'has_garden', 'has_porter',
    'has_gym', 'has_pool', 'has_parking', 'has_lift', 'has_ac',
    'has_furnished', 'has_high_ceilings', 'has_view',
    'has_modern', 'has_period', 'has_roof_terrace'
]


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_nearest_tube_distance(lat, lon):
    if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
        return None
    return min(haversine_distance(lat, lon, slat, slon) for slat, slon in TUBE_STATIONS.values())


def get_distance_to_center(lat, lon):
    if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
        return None
    return haversine_distance(lat, lon, CITY_CENTER[0], CITY_CENTER[1])


def parse_amenities(features_str, description_str=''):
    """Extract amenity flags from features JSON and description text."""
    amenities = {f: 0 for f in AMENITY_FEATURES}

    text = ''
    if features_str and not pd.isna(features_str):
        text += str(features_str).lower() + ' '
    if description_str and not pd.isna(description_str):
        text += str(description_str).lower()

    # Parse JSON
    try:
        if features_str and not pd.isna(features_str):
            parsed = json.loads(features_str)
            if isinstance(parsed, dict):
                for key in AMENITY_FEATURES:
                    if parsed.get(key):
                        amenities[key] = 1
    except (json.JSONDecodeError, TypeError):
        pass

    # Text-based extraction
    amenities['has_balcony'] = max(amenities['has_balcony'], int('balcony' in text))
    amenities['has_terrace'] = max(amenities['has_terrace'], int('terrace' in text and 'roof terrace' not in text))
    amenities['has_roof_terrace'] = max(amenities['has_roof_terrace'], int('roof terrace' in text))
    amenities['has_garden'] = max(amenities['has_garden'], int('garden' in text))
    amenities['has_porter'] = max(amenities['has_porter'], int('porter' in text or 'concierge' in text))
    amenities['has_gym'] = max(amenities['has_gym'], int('gym' in text))
    amenities['has_pool'] = max(amenities['has_pool'], int('pool' in text or 'swimming' in text))
    amenities['has_parking'] = max(amenities['has_parking'], int('parking' in text or 'garage' in text))
    amenities['has_lift'] = max(amenities['has_lift'], int('lift' in text or 'elevator' in text))
    amenities['has_ac'] = max(amenities['has_ac'], int('air con' in text or 'a/c' in text or 'aircon' in text))
    amenities['has_high_ceilings'] = max(amenities['has_high_ceilings'], int('high ceiling' in text))
    amenities['has_view'] = max(amenities['has_view'], int('view' in text))
    amenities['has_modern'] = max(amenities['has_modern'], int('modern' in text or 'contemporary' in text))
    amenities['has_period'] = max(amenities['has_period'], int('period' in text or 'victorian' in text or 'georgian' in text))

    return amenities


def load_and_clean_data():
    """Load data from Postgres (if POSTGRES_URL set) or SQLite."""
    import os

    query = """
        SELECT
            bedrooms, bathrooms, size_sqft,
            postcode, area, property_type,
            price_pcm, latitude, longitude, features, description, source,
            property_type_std, let_type, postcode_normalized,
            postcode_inferred, agent_brand,
            floor_count, has_roof_terrace, has_basement, has_ground,
            has_first_floor, has_second_floor, has_third_floor, has_fourth_plus
        FROM listings
        WHERE size_sqft > 0 AND bedrooms IS NOT NULL AND price_pcm > 0
        AND is_active = 1
        AND (is_short_let = 0 OR is_short_let IS NULL)
    """

    postgres_url = os.environ.get('POSTGRES_URL')
    if postgres_url:
        import psycopg2
        conn = psycopg2.connect(postgres_url)
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Loaded from Postgres")
    else:
        conn = sqlite3.connect('output/rentals.db')
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Loaded from SQLite")

    print(f"Raw data: {len(df)} records")

    # Quality filters (no price-based filters except obvious outliers)
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    mask1 = sqft_per_bed < 70  # Too small to be real
    mask2 = df['price_pcm'] > 100000  # Obvious data error
    mask3 = df['size_sqft'] < 150  # Studio minimum
    mask4 = df['size_sqft'] > 20000  # Obvious error

    df_clean = df[~(mask1 | mask2 | mask3 | mask4)].copy()
    print(f"Clean data: {len(df_clean)} records")

    # === DEDUPE: Remove cross-source duplicates before training ===
    # Same property appears on Rightmove AND agent sites - keep agent version (better data)
    # Match by: price + size + bedrooms + postcode district
    df_clean['postcode_district'] = df_clean['postcode'].str.extract(r'^([A-Z]+\d+[A-Z]?)', expand=False)

    # Source priority: agents first (better data), rightmove last
    source_priority = {'savills': 0, 'knightfrank': 1, 'chestertons': 2, 'foxtons': 3, 'rightmove': 4}
    df_clean['source_rank'] = df_clean['source'].map(source_priority).fillna(5)

    # Sort by source rank (agents first), then drop duplicates keeping first (best source)
    df_clean = df_clean.sort_values('source_rank')
    before_dedupe = len(df_clean)
    df_clean = df_clean.drop_duplicates(
        subset=['price_pcm', 'size_sqft', 'bedrooms', 'postcode_district'],
        keep='first'
    )
    dupes_removed = before_dedupe - len(df_clean)
    print(f"Deduped data: {len(df_clean)} records ({dupes_removed} cross-source duplicates removed)")

    # Clean up temp columns
    df_clean = df_clean.drop(columns=['source_rank'])

    return df_clean


def engineer_features_v15(df):
    """V15 feature engineering - NO LEAKAGE."""
    print("\n[FEATURE ENGINEERING V15 - NO LEAKAGE]")

    # ========== AMENITY FEATURES (from text, no price info) ==========
    amenity_dicts = df.apply(lambda r: parse_amenities(r['features'], r.get('description', '')), axis=1)
    amenity_df = pd.DataFrame(amenity_dicts.tolist())
    for col in AMENITY_FEATURES:
        df[col] = amenity_df[col].values

    df['amenity_score'] = df[AMENITY_FEATURES].sum(axis=1)
    df['luxury_amenity_score'] = df[['has_pool', 'has_porter', 'has_gym', 'has_ac']].sum(axis=1)
    df['has_outdoor_space'] = ((df['has_balcony'] == 1) | (df['has_terrace'] == 1) |
                               (df['has_garden'] == 1) | (df['has_roof_terrace'] == 1)).astype(int)
    print(f"  Amenity score: mean={df['amenity_score'].mean():.2f}")

    # ========== PROPERTY TYPE (categorical, no price info) ==========
    df['property_type_std'] = df['property_type_std'].fillna('flat')
    type_map = {'studio': 0, 'flat': 1, 'apartment': 1, 'maisonette': 2, 'house': 3, 'penthouse': 4}
    df['property_type_num'] = df['property_type_std'].map(type_map).fillna(1)

    # ========== LET TYPE (categorical) ==========
    df['let_type'] = df['let_type'].fillna('unknown')
    df['is_short_let'] = (df['let_type'] == 'short').astype(int)
    df['is_long_let'] = (df['let_type'] == 'long').astype(int)

    # ========== POSTCODE (location-based, no price info) ==========
    df['postcode_district'] = df['postcode_normalized'].fillna('SW3')
    df['postcode_area'] = df['postcode_district'].str.extract(r'^([A-Z]+)', expand=False).fillna('SW')
    df['is_prime_postcode'] = df['postcode_district'].apply(
        lambda x: int(any(x.startswith(p) for p in PRIME_POSTCODES))
    )
    print(f"  Prime postcodes: {df['is_prime_postcode'].sum()} ({100*df['is_prime_postcode'].mean():.1f}%)")

    # ========== AGENT BRAND (categorical, no price info) ==========
    df['agent_brand'] = df['agent_brand'].fillna('unknown')
    df['is_premium_agent'] = df['agent_brand'].isin(PREMIUM_AGENTS).astype(int)
    print(f"  Premium agents: {df['is_premium_agent'].sum()} ({100*df['is_premium_agent'].mean():.1f}%)")

    # ========== COORDINATES & DISTANCES ==========
    def get_coords(row):
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or lat == 0:
            pc = row['postcode_district']
            if pc in POSTCODE_CENTROIDS:
                return pd.Series(POSTCODE_CENTROIDS[pc])
            for key in POSTCODE_CENTROIDS:
                if pc.startswith(key):
                    return pd.Series(POSTCODE_CENTROIDS[key])
        return pd.Series([lat, lon])

    df[['lat_filled', 'lon_filled']] = df.apply(get_coords, axis=1)
    df['tube_distance_km'] = df.apply(lambda r: get_nearest_tube_distance(r['lat_filled'], r['lon_filled']), axis=1)
    df['center_distance_km'] = df.apply(lambda r: get_distance_to_center(r['lat_filled'], r['lon_filled']), axis=1)

    median_tube = df['tube_distance_km'].median()
    median_center = df['center_distance_km'].median()
    df['tube_distance_km'] = df['tube_distance_km'].fillna(median_tube)
    df['center_distance_km'] = df['center_distance_km'].fillna(median_center)

    # ========== SIZE & ROOM FEATURES ==========
    df['bathrooms'] = df['bathrooms'].fillna(1)
    beds_adj = df['bedrooms'].replace(0, 0.5)

    df['size_per_bed'] = df['size_sqft'] / beds_adj
    df['bath_ratio'] = df['bathrooms'] / beds_adj
    df['bed_bath_interaction'] = df['bedrooms'] * df['bathrooms']
    df['log_sqft'] = np.log1p(df['size_sqft'])
    df['sqrt_sqft'] = np.sqrt(df['size_sqft'])
    df['size_squared'] = df['size_sqft'] ** 2 / 100000
    df['beds_squared'] = df['bedrooms'] ** 2

    # Size bins (quintiles of size, not price)
    df['size_bin'] = pd.qcut(df['size_sqft'], q=5, labels=[0, 1, 2, 3, 4], duplicates='drop').astype(float).fillna(2)

    # ========== FLOOR FEATURES ==========
    df['floor_count'] = df['floor_count'].fillna(0)
    df['has_basement'] = df['has_basement'].fillna(0)
    df['has_ground'] = df['has_ground'].fillna(0)
    df['has_first_floor'] = df['has_first_floor'].fillna(0)
    df['has_second_floor'] = df['has_second_floor'].fillna(0)
    df['has_third_floor'] = df['has_third_floor'].fillna(0)
    df['has_fourth_plus'] = df['has_fourth_plus'].fillna(0)

    df['is_multi_floor'] = (df['floor_count'] >= 2).astype(int)
    df['floor_size_interaction'] = df['floor_count'] * df['size_sqft'] / 1000
    print(f"  Multi-floor properties: {df['is_multi_floor'].sum()} ({100*df['is_multi_floor'].mean():.1f}%)")

    # ========== LOCATION INTERACTIONS (no price info) ==========
    df['log_center_distance'] = np.log1p(df['center_distance_km'])
    df['log_tube_distance'] = np.log1p(df['tube_distance_km'])
    df['center_distance_inv'] = 1 / (1 + df['center_distance_km'])

    # Size x location interactions
    df['size_x_central'] = df['size_sqft'] * df['center_distance_inv'] / 100
    df['size_x_prime'] = df['size_sqft'] * df['is_prime_postcode'] / 1000
    df['beds_x_central'] = df['bedrooms'] * df['center_distance_inv']

    # Amenity x location interactions
    df['amenity_x_central'] = df['amenity_score'] * df['center_distance_inv']
    df['outdoor_x_prime'] = df['has_outdoor_space'] * df['is_prime_postcode']

    # Agent x size interactions
    df['premium_agent_size'] = df['is_premium_agent'] * df['log_sqft']

    # Let type interactions
    df['short_let_x_central'] = df['is_short_let'] * df['center_distance_inv']
    df['short_let_size'] = df['is_short_let'] * df['log_sqft']

    # ========== LABEL ENCODE CATEGORICALS ==========
    # Postcode district (frequency encoding - no price info)
    pc_counts = df['postcode_district'].value_counts()
    df['postcode_freq'] = df['postcode_district'].map(pc_counts) / len(df)

    # Postcode area encoding
    area_counts = df['postcode_area'].value_counts()
    df['postcode_area_freq'] = df['postcode_area'].map(area_counts) / len(df)

    # Source encoding
    source_map = {'savills': 4, 'knightfrank': 4, 'chestertons': 3, 'foxtons': 2, 'rightmove': 1, 'johndwood': 3}
    df['source_quality'] = df['source'].map(source_map).fillna(2)

    # ========== ONE-HOT ENCODE PROPERTY TYPE ==========
    df = pd.get_dummies(df, columns=['property_type_std'], prefix='type')

    # ========== ONE-HOT ENCODE TOP POSTCODES ==========
    top_postcodes = df['postcode_district'].value_counts().head(15).index.tolist()
    for pc in top_postcodes:
        df[f'pc_{pc}'] = (df['postcode_district'] == pc).astype(int)

    total_features = len([c for c in df.columns if c not in ['price_pcm', 'features', 'description', 'postcode', 'area', 'property_type', 'postcode_normalized', 'agent_brand', 'let_type', 'postcode_district', 'postcode_area', 'source', 'latitude', 'longitude', 'lat_filled', 'lon_filled', 'postcode_inferred']])
    print(f"  Total features: {total_features}")

    return df


def get_feature_columns(df):
    """Get list of features for model (no leaky features)."""
    feature_cols = [
        # Core features
        'bedrooms', 'bathrooms', 'size_sqft',
        'size_per_bed', 'bath_ratio', 'bed_bath_interaction',
        'log_sqft', 'sqrt_sqft', 'size_squared', 'beds_squared', 'size_bin',

        # Location features (no price encoding)
        'tube_distance_km', 'log_tube_distance',
        'center_distance_km', 'log_center_distance', 'center_distance_inv',
        'is_prime_postcode', 'postcode_freq', 'postcode_area_freq',

        # Size x location interactions
        'size_x_central', 'size_x_prime', 'beds_x_central',

        # Floor features
        'floor_count', 'is_multi_floor', 'floor_size_interaction',
        'has_basement', 'has_ground', 'has_first_floor',
        'has_second_floor', 'has_third_floor', 'has_fourth_plus',

        # Agent/source
        'is_premium_agent', 'premium_agent_size', 'source_quality',

        # Amenity features
        'amenity_score', 'luxury_amenity_score', 'has_outdoor_space',
        'amenity_x_central', 'outdoor_x_prime',

        # Let type
        'is_short_let', 'is_long_let',
        'short_let_x_central', 'short_let_size',

        # Property type encoding
        'property_type_num',
    ]

    # Add amenity columns
    feature_cols.extend(AMENITY_FEATURES)

    # Add one-hot encoded columns
    feature_cols.extend([c for c in df.columns if c.startswith('type_')])
    feature_cols.extend([c for c in df.columns if c.startswith('pc_')])

    # Filter to columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    return feature_cols


def evaluate_model(model, X, y, cv=5):
    """Evaluate with cross-validation."""
    y_log = np.log1p(y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    all_preds = []
    all_actual = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = np.expm1(model.predict(X_val))

        all_preds.extend(y_pred)
        all_actual.extend(y_val.values)

    all_preds = np.array(all_preds)
    all_actual = np.array(all_actual)

    return {
        'MAE': mean_absolute_error(all_actual, all_preds),
        'RMSE': np.sqrt(mean_squared_error(all_actual, all_preds)),
        'R2': r2_score(all_actual, all_preds),
        'MAPE': np.mean(np.abs((all_actual - all_preds) / all_actual)) * 100,
        'Median_APE': np.median(np.abs((all_actual - all_preds) / all_actual)) * 100,
        'predictions': all_preds,
        'actual': all_actual
    }


def build_xgboost():
    """Build XGBoost with good defaults."""
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
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )


def tune_optuna(X, y, n_trials=30):
    """Tune with Optuna."""
    if not HAS_OPTUNA:
        return {}

    print(f"  Running Optuna ({n_trials} trials)...")
    y_log = np.log1p(y)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        }

        model = XGBRegressor(
            **params,
            objective='reg:absoluteerror',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model.fit(X.iloc[train_idx], y_log.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores.append(r2_score(y_log.iloc[val_idx], preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best R² (log): {study.best_value:.4f}")
    return study.best_params


def analyze_errors_by_tier(actual, preds):
    tiers = [
        (0, 3000, '<£3k'),
        (3000, 5000, '£3-5k'),
        (5000, 10000, '£5-10k'),
        (10000, 20000, '£10-20k'),
        (20000, 100000, '£20k+')
    ]

    print("\n  Error by Price Tier:")
    for low, high, label in tiers:
        mask = (actual >= low) & (actual < high)
        if mask.sum() > 0:
            tier_preds = preds[mask]
            tier_actual = actual[mask]
            tier_mape = np.mean(np.abs((tier_actual - tier_preds) / tier_actual)) * 100
            tier_mae = np.mean(np.abs(tier_actual - tier_preds))
            print(f"    {label}: n={mask.sum()}, MAPE={tier_mape:.1f}%, MAE=£{tier_mae:,.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Skip Optuna tuning')
    args = parser.parse_args()

    print("=" * 70)
    print("LONDON RENTAL PRICE MODELS V15 - NO LEAKAGE")
    print("=" * 70)
    print("Removed: is_luxury, ppsf_encoded, target mean encoding")
    print("Using: location features, amenities, property characteristics only")

    df = load_and_clean_data()
    df = engineer_features_v15(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0)
    y = df['price_pcm']

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")

    results = {}

    # ========== XGBoost Default ==========
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    print("\n[XGBoost_V15_Default]")
    model = build_xgboost()
    result = evaluate_model(model, X, y)
    results['XGBoost_V15_Default'] = result
    print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
    analyze_errors_by_tier(result['actual'], result['predictions'])

    # ========== Optuna Tuning ==========
    if not args.quick and HAS_OPTUNA:
        print("\n[XGBoost_V15_Optuna]")
        best_params = tune_optuna(X, y, n_trials=30)
        if best_params:
            model = XGBRegressor(
                **best_params,
                objective='reg:absoluteerror',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            result = evaluate_model(model, X, y)
            results['XGBoost_V15_Optuna'] = result
            print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
            analyze_errors_by_tier(result['actual'], result['predictions'])

    # ========== Best Model ==========
    best_name = max(results, key=lambda x: results[x]['R2'])
    best = results[best_name]

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}")
    print("=" * 70)
    print(f"R²:         {best['R2']:.4f}")
    print(f"MAE:        £{best['MAE']:,.0f}")
    print(f"MAPE:       {best['MAPE']:.1f}%")
    print(f"Median APE: {best['Median_APE']:.1f}%")

    # ========== Feature Importance ==========
    print("\n" + "-" * 70)
    print("TOP 25 FEATURE IMPORTANCE")
    print("-" * 70)

    final_model = build_xgboost()
    final_model.fit(X, np.log1p(y))

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(25).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:<35} {row['importance']:.3f} {bar}")

    # ========== Verify No Leakage ==========
    print("\n" + "-" * 70)
    print("LEAKAGE CHECK")
    print("-" * 70)

    leaky_keywords = ['luxury', 'ppsf', 'price', '_encoded', 'is_luxury']
    leaky_features = [f for f in feature_cols if any(kw in f.lower() for kw in leaky_keywords)]

    if leaky_features:
        print(f"  WARNING: Potential leaky features found: {leaky_features}")
    else:
        print("  PASSED: No target-derived features detected")

    # ========== Save Model ==========
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / 'rental_model_v15.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open(OUTPUT_DIR / 'rental_model_v15_features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"\nSaved model to {OUTPUT_DIR / 'rental_model_v15.pkl'}")

    # ========== Version History ==========
    print("\n" + "=" * 70)
    print("VERSION COMPARISON (V15 = No Leakage)")
    print("=" * 70)
    print("Historical results (WITH leakage):")
    print("  V6: R²=0.791, V7: R²=0.786, V9: R²=0.793")
    print("  V14: R²=0.674")
    print(f"\nV15 (NO leakage): R²={best['R2']:.3f}, MAE=£{best['MAE']:,.0f}, MAPE={best['MAPE']:.1f}%")

    # ========== Log to Postgres (if available) ==========
    log_model_metrics_to_postgres(best, len(X), len(feature_cols))

    return results


def log_model_metrics_to_postgres(metrics, samples, features_count):
    """Log model run metrics to Postgres model_runs table."""
    import os
    from datetime import datetime

    postgres_url = os.environ.get('POSTGRES_URL')
    if not postgres_url:
        print("\nSkipping Postgres logging (POSTGRES_URL not set)")
        return

    try:
        import psycopg2
        conn = psycopg2.connect(postgres_url)
        cur = conn.cursor()

        run_date = datetime.utcnow().strftime('%Y-%m-%d')
        run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        cur.execute('''
            INSERT INTO model_runs (
                run_date, run_id, version, samples_total, features_count,
                r2_score, mae, mape, median_ape
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            run_date, run_id, 'v15', samples, features_count,
            metrics['R2'], metrics['MAE'], metrics['MAPE'], metrics['Median_APE']
        ))

        conn.commit()
        conn.close()
        print(f"\n✅ Logged model metrics to Postgres (run_id: {run_id})")
    except Exception as e:
        print(f"\n⚠️ Failed to log to Postgres: {e}")


if __name__ == '__main__':
    results = main()
