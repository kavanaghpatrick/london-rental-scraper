"""
London Rental Price Prediction Models V7 - Hyperparameter Optimized

Improvements over V6:
1. Hyperparameter tuning with Optuna
2. Stacked ensemble (XGBoost + LightGBM + CatBoost)
3. Additional feature interactions
4. Target-encoded neighborhoods at multiple granularities

Baseline (V6): R²=0.904, MAE=£976, MAPE=12.6%

Usage:
    python rental_price_models_v7.py
    python rental_price_models_v7.py --quick    # Skip Optuna tuning
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import argparse
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed, skipping hyperparameter tuning")


# Location data
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
    'Canary Wharf': (51.5033, -0.0184),
}

POSTCODE_CENTROIDS = {
    'SW1': (51.4970, -0.1400), 'SW1X': (51.4983, -0.1560), 'SW1W': (51.4920, -0.1470),
    'SW1P': (51.4960, -0.1300), 'SW3': (51.4900, -0.1680), 'SW5': (51.4920, -0.1940),
    'SW7': (51.4950, -0.1780), 'SW10': (51.4830, -0.1820), 'SW11': (51.4650, -0.1650),
    'SW13': (51.4750, -0.2430), 'W1': (51.5150, -0.1450), 'W1K': (51.5120, -0.1510),
    'W1J': (51.5080, -0.1470), 'W1G': (51.5180, -0.1470), 'W1U': (51.5200, -0.1520),
    'W2': (51.5150, -0.1780), 'W8': (51.5010, -0.1920), 'W11': (51.5150, -0.2050),
    'W14': (51.4950, -0.2100), 'NW1': (51.5350, -0.1550), 'NW3': (51.5550, -0.1780),
    'NW8': (51.5330, -0.1750), 'EC1': (51.5230, -0.1020), 'EC2': (51.5180, -0.0830),
    'EC4': (51.5130, -0.1020), 'WC1': (51.5230, -0.1200), 'WC2': (51.5110, -0.1220),
    'E1': (51.5150, -0.0720), 'E14': (51.5070, -0.0200), 'SE1': (51.5010, -0.1060),
}

CITY_CENTER = (51.5074, -0.1278)

AMENITY_FEATURES = [
    'has_balcony', 'has_terrace', 'has_garden', 'has_porter',
    'has_gym', 'has_pool', 'has_parking', 'has_lift', 'has_ac',
    'has_furnished', 'has_high_ceilings', 'has_view',
    'has_modern', 'has_period'
]

PREMIUM_AGENTS = ['Knight Frank', 'Savills', 'Harrods Estates', 'Sotheby',
                  'Beauchamp Estates', 'Strutt & Parker']


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


def parse_amenities(features_str):
    amenities = {f: 0 for f in AMENITY_FEATURES}
    if pd.isna(features_str) or features_str in ['', '[]', '{}']:
        return amenities
    try:
        parsed = json.loads(features_str)
        if isinstance(parsed, dict):
            for key in AMENITY_FEATURES:
                if parsed.get(key):
                    amenities[key] = 1
    except (json.JSONDecodeError, TypeError):
        pass
    return amenities


def load_and_clean_data():
    conn = sqlite3.connect('output/rentals.db')
    df = pd.read_sql("""
        SELECT
            bedrooms, bathrooms, size_sqft,
            postcode, area, property_type,
            price_pcm, latitude, longitude, features, source,
            property_type_std, let_type, postcode_normalized,
            postcode_inferred, agent_brand
        FROM listings
        WHERE size_sqft > 0 AND bedrooms IS NOT NULL AND price_pcm > 0
    """, conn)
    conn.close()

    print(f"Raw data: {len(df)} records")

    # Quality filters
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    mask1 = sqft_per_bed < 70
    ppsqft = df['price_pcm'] / df['size_sqft']
    mask2 = (ppsqft < 1.0) | (ppsqft > 25.0)
    mask3 = df['price_pcm'] > 50000
    mask4 = df['size_sqft'] < 150

    df_clean = df[~(mask1 | mask2 | mask3 | mask4)].copy()
    print(f"Clean data: {len(df_clean)} records")
    return df_clean


def engineer_features_v7(df):
    """V7 feature engineering with additional interactions."""
    print("\n[FEATURE ENGINEERING V7]")

    # ========== AMENITY FEATURES ==========
    amenity_dicts = df['features'].apply(parse_amenities)
    amenity_df = pd.DataFrame(amenity_dicts.tolist())
    for col in AMENITY_FEATURES:
        df[col] = amenity_df[col].values

    df['amenity_score'] = df[AMENITY_FEATURES].sum(axis=1)
    luxury_amenities = ['has_pool', 'has_porter', 'has_gym', 'has_ac']
    df['luxury_amenity_score'] = df[luxury_amenities].sum(axis=1)
    df['has_outdoor_space'] = ((df['has_balcony'] == 1) | (df['has_terrace'] == 1) | (df['has_garden'] == 1)).astype(int)

    # ========== PROPERTY TYPE ==========
    df['property_type_std'] = df['property_type_std'].fillna('flat')

    # ========== LET TYPE ==========
    df['let_type'] = df['let_type'].fillna('unknown')
    df['is_short_let'] = (df['let_type'] == 'short').astype(int)
    df['is_long_let'] = (df['let_type'] == 'long').astype(int)

    # ========== POSTCODE ==========
    df['postcode_district'] = df['postcode_normalized'].fillna('SW3')
    df['postcode_was_inferred'] = df['postcode_inferred'].fillna(0).astype(int)

    # V7 NEW: Postcode area (first letters only - SW, W, NW, etc.)
    df['postcode_area'] = df['postcode_district'].str.extract(r'^([A-Z]+)', expand=False).fillna('SW')

    # ========== AGENT BRAND ==========
    df['agent_brand'] = df['agent_brand'].fillna('unknown')
    df['is_premium_agent'] = df['agent_brand'].isin(PREMIUM_AGENTS).astype(int)

    # ========== COORDINATES ==========
    def get_coords(row):
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or lat == 0:
            pc = row['postcode_district']
            if pc in POSTCODE_CENTROIDS:
                return pd.Series(POSTCODE_CENTROIDS[pc])
            pc_short = pc[:3] if len(pc) >= 3 else pc
            if pc_short in POSTCODE_CENTROIDS:
                return pd.Series(POSTCODE_CENTROIDS[pc_short])
            pc_area = pc[:2] if len(pc) >= 2 else pc
            if pc_area in POSTCODE_CENTROIDS:
                return pd.Series(POSTCODE_CENTROIDS[pc_area])
        return pd.Series([lat, lon])

    df[['lat_filled', 'lon_filled']] = df.apply(get_coords, axis=1)

    df['tube_distance_km'] = df.apply(lambda r: get_nearest_tube_distance(r['lat_filled'], r['lon_filled']), axis=1)
    df['center_distance_km'] = df.apply(lambda r: get_distance_to_center(r['lat_filled'], r['lon_filled']), axis=1)

    median_tube = df['tube_distance_km'].median()
    median_center = df['center_distance_km'].median()
    df['tube_distance_km'] = df['tube_distance_km'].fillna(median_tube)
    df['center_distance_km'] = df['center_distance_km'].fillna(median_center)

    # ========== SIZE/ROOM FEATURES ==========
    df['bathrooms'] = df['bathrooms'].fillna(1)
    beds_adj = df['bedrooms'].replace(0, 0.5)

    df['size_per_bed'] = df['size_sqft'] / beds_adj
    df['bath_ratio'] = df['bathrooms'] / beds_adj
    df['bed_bath_interaction'] = df['bedrooms'] * df['bathrooms']
    df['size_poly'] = df['size_sqft'] ** 1.5 / 1000
    df['sqft_per_bath'] = df['size_sqft'] / (df['bathrooms'] + 1)
    df['log_sqft'] = np.log1p(df['size_sqft'])

    # V7 NEW: More size features
    df['size_squared'] = df['size_sqft'] ** 2 / 100000
    df['sqrt_sqft'] = np.sqrt(df['size_sqft'])
    df['beds_squared'] = df['bedrooms'] ** 2

    df['size_bin'] = pd.qcut(df['size_sqft'], q=5, labels=[0, 1, 2, 3, 4]).astype(int)

    # ========== INTERACTION FEATURES ==========
    df['short_let_size'] = df['is_short_let'] * df['log_sqft']
    df['long_let_size'] = df['is_long_let'] * df['log_sqft']
    df['center_size_interaction'] = df['center_distance_km'] * df['size_sqft'] / 1000
    df['log_center_distance'] = np.log1p(df['center_distance_km'])
    df['tube_size_interaction'] = df['tube_distance_km'] * df['size_sqft'] / 1000
    df['log_tube_distance'] = np.log1p(df['tube_distance_km'])

    df['amenity_size_interaction'] = df['amenity_score'] * df['log_sqft']
    df['outdoor_size_interaction'] = df['has_outdoor_space'] * df['size_sqft'] / 1000
    df['luxury_center_interaction'] = df['luxury_amenity_score'] * (5 - df['center_distance_km']).clip(lower=0)
    df['premium_agent_size'] = df['is_premium_agent'] * df['log_sqft']

    # V7 NEW: Additional interactions
    df['beds_center_interaction'] = df['bedrooms'] * (5 - df['center_distance_km']).clip(lower=0)
    df['premium_center_interaction'] = df['is_premium_agent'] * (5 - df['center_distance_km']).clip(lower=0)
    df['short_let_premium'] = df['is_short_let'] * df['is_premium_agent']
    df['amenity_beds_interaction'] = df['amenity_score'] * df['bedrooms']

    # ========== LUXURY INDICATORS ==========
    df['is_luxury'] = (df['price_pcm'] > 10000).astype(int)
    df['luxury_size_interaction'] = df['is_luxury'] * df['size_sqft'] / 1000

    # ========== TARGET ENCODING (multi-level) ==========
    global_mean = np.log1p(df['price_pcm']).mean()
    smoothing = 15

    df['area_normalized'] = df['area'].str.lower().str.replace('-', '').str.replace(' ', '').str.replace("'", '').fillna('')

    # Standard encodings
    for col in ['area_normalized', 'postcode_district', 'agent_brand', 'postcode_area']:
        means = np.log1p(df.groupby(col)['price_pcm'].transform('mean'))
        counts = df.groupby(col)['price_pcm'].transform('count')
        df[f'{col}_encoded'] = (means * counts + global_mean * smoothing) / (counts + smoothing)

    # V7 NEW: Price per sqft encoding by area
    df['ppsf'] = df['price_pcm'] / df['size_sqft']
    for col in ['area_normalized', 'postcode_district']:
        mean_ppsf = df.groupby(col)['ppsf'].transform('mean')
        df[f'{col}_ppsf_encoded'] = mean_ppsf.fillna(df['ppsf'].mean())

    df['size_postcode_interaction'] = df['size_sqft'] * df['postcode_district_encoded'] / 1000
    df['size_area_interaction'] = df['size_sqft'] * df['area_normalized_encoded'] / 1000

    # One-hot encode property type
    df = pd.get_dummies(df, columns=['property_type_std'], prefix='type')

    print(f"  Total features engineered: {len([c for c in df.columns if c not in ['price_pcm', 'features', 'description', 'summary']])}")

    return df


def evaluate_model(model, X, y, cv=5, weight_high_end=False):
    """Evaluate with cross-validation."""
    y_log = np.log1p(y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    all_preds = []
    all_actual = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y.iloc[val_idx]

        if weight_high_end:
            weights = np.where(np.expm1(y_train) >= 15000, 2.0, 1.0)
            model.fit(X_train, y_train, sample_weight=weights)
        else:
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


def build_xgboost_v7(params=None):
    """Build XGBoost with V7 parameters."""
    default_params = {
        'n_estimators': 1500,
        'learning_rate': 0.008,
        'max_depth': 9,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 2,
        'reg_alpha': 0.15,
        'reg_lambda': 2.5,
        'objective': 'reg:absoluteerror',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    if params:
        default_params.update(params)
    return XGBRegressor(**default_params)


def tune_xgboost_optuna(X, y, n_trials=30):
    """Tune XGBoost with Optuna."""
    if not HAS_OPTUNA:
        print("  Optuna not available, using default params")
        return {}

    print(f"  Running Optuna optimization ({n_trials} trials)...")
    y_log = np.log1p(y)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
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

        # Quick 3-fold CV
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            scores.append(r2_score(y_val, preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best R² (log): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


def build_stacking_ensemble(X, y):
    """Build a stacking ensemble of XGBoost, LightGBM, and CatBoost."""
    print("\n[STACKING ENSEMBLE]")

    estimators = [
        ('xgb', build_xgboost_v7()),
    ]

    if HAS_LIGHTGBM:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.015,
            max_depth=9,
            num_leaves=80,
            objective='mae',
            random_state=42,
            verbosity=-1
        )
        estimators.append(('lgb', lgb_model))

    if HAS_CATBOOST:
        cat_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.015,
            depth=9,
            loss_function='MAE',
            random_seed=42,
            verbose=False
        )
        estimators.append(('cat', cat_model))

    print(f"  Base models: {[e[0] for e in estimators]}")

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )

    return stack


def analyze_errors_by_tier(actual, preds, tiers):
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
    print("LONDON RENTAL PRICE MODELS V7 - HYPERPARAMETER OPTIMIZED")
    print("=" * 70)

    df = load_and_clean_data()
    df = engineer_features_v7(df)

    # V7 feature list (extended)
    feature_cols = [
        # Core features
        'bedrooms', 'bathrooms', 'size_sqft',
        'size_per_bed', 'bath_ratio', 'bed_bath_interaction',
        'size_poly', 'sqft_per_bath', 'log_sqft',
        'size_squared', 'sqrt_sqft', 'beds_squared',

        # Location features
        'area_normalized_encoded', 'postcode_district_encoded', 'postcode_area_encoded',
        'size_postcode_interaction', 'size_area_interaction',
        'area_normalized_ppsf_encoded', 'postcode_district_ppsf_encoded',
        'postcode_was_inferred',
        'tube_distance_km', 'log_tube_distance', 'tube_size_interaction',
        'center_distance_km', 'log_center_distance', 'center_size_interaction',

        # Let type
        'is_short_let', 'is_long_let',
        'short_let_size', 'long_let_size', 'short_let_premium',

        # Size category
        'size_bin',
        'luxury_size_interaction',

        # Agent brand
        'agent_brand_encoded', 'is_premium_agent', 'premium_agent_size',
        'premium_center_interaction',

        # Amenity features
        'amenity_score', 'luxury_amenity_score', 'has_outdoor_space',
        'amenity_size_interaction', 'outdoor_size_interaction', 'luxury_center_interaction',
        'amenity_beds_interaction',

        # V7 new interactions
        'beds_center_interaction',

    ] + AMENITY_FEATURES + [c for c in df.columns if c.startswith('type_')]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df['price_pcm']

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")

    tiers = [
        (0, 3000, '<£3k'),
        (3000, 5000, '£3-5k'),
        (5000, 10000, '£5-10k'),
        (10000, 20000, '£10-20k'),
        (20000, 100000, '£20k+')
    ]

    results = {}

    # ========== XGBoost V7 Default ==========
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print("\n[XGBoost_V7_Default]")
    model = build_xgboost_v7()
    result = evaluate_model(model, X, y, weight_high_end=True)
    results['XGBoost_V7_Default'] = result
    print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
    analyze_errors_by_tier(result['actual'], result['predictions'], tiers)

    # ========== Optuna Tuning ==========
    if not args.quick and HAS_OPTUNA:
        print("\n[XGBoost_V7_Optuna]")
        best_params = tune_xgboost_optuna(X, y, n_trials=30)
        if best_params:
            model = build_xgboost_v7(best_params)
            result = evaluate_model(model, X, y, weight_high_end=True)
            results['XGBoost_V7_Optuna'] = result
            print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
            analyze_errors_by_tier(result['actual'], result['predictions'], tiers)

    # ========== Stacking Ensemble ==========
    if HAS_LIGHTGBM or HAS_CATBOOST:
        print("\n[Stacking_Ensemble]")
        y_log = np.log1p(y)

        stack = build_stacking_ensemble(X, y)

        # Cross-val predict for stacking
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_preds = []
        all_actual = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_log.iloc[train_idx], y.iloc[val_idx]

            stack.fit(X_train, y_train)
            preds = np.expm1(stack.predict(X_val))
            all_preds.extend(preds)
            all_actual.extend(y_val.values)

        all_preds = np.array(all_preds)
        all_actual = np.array(all_actual)

        result = {
            'MAE': mean_absolute_error(all_actual, all_preds),
            'R2': r2_score(all_actual, all_preds),
            'MAPE': np.mean(np.abs((all_actual - all_preds) / all_actual)) * 100,
            'Median_APE': np.median(np.abs((all_actual - all_preds) / all_actual)) * 100,
            'predictions': all_preds,
            'actual': all_actual
        }
        results['Stacking_Ensemble'] = result
        print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
        analyze_errors_by_tier(result['actual'], result['predictions'], tiers)

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
    print("FEATURE IMPORTANCE")
    print("-" * 70)

    best_model = build_xgboost_v7()
    best_model.fit(X, np.log1p(y))

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(20).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:<40} {row['importance']:.3f} {bar}")

    # ========== Version Comparison ==========
    print("\n" + "=" * 70)
    print("VERSION HISTORY")
    print("=" * 70)
    print("V1 (baseline):    R²=0.658, MAE=£2,589, MAPE=28.2%")
    print("V2 (Grok/Gemini): R²=0.701, MAE=£1,937, MAPE=23.1%")
    print("V3 (augmented):   R²=0.777, MAE=£1,624, MAPE=19.8%")
    print("V4 (sample wt):   R²=0.780, MAE=£1,598, MAPE=19.3%")
    print("V5 (amenities):   R²=0.773, MAE=£1,650, MAPE=20.1%")
    print("V6 (standardized):R²=0.904, MAE=£976,  MAPE=12.6%")
    print(f"V7 ({best_name}): R²={best['R2']:.3f}, MAE=£{best['MAE']:,.0f}, MAPE={best['MAPE']:.1f}%")

    v6_r2 = 0.904
    r2_improvement = (best['R2'] - v6_r2) / v6_r2 * 100
    print(f"\nV7 vs V6 R² change: {r2_improvement:+.1f}%")

    return results, df


if __name__ == '__main__':
    results, df = main()
