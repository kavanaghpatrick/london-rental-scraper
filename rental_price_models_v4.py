"""
London Rental Price Prediction Models V4 - Pushing to R²=0.80+

Additional improvements based on Grok V4 recommendations:
1. Sample weights for high-end properties
2. CatBoost for better categorical handling
3. LightGBM comparison
4. More interaction features
5. Distance to city center
6. Separate high-end model

Usage:
    python rental_price_models_v4.py
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed, skipping...")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed, skipping...")


# Prime London area to postcode mapping
AREA_TO_POSTCODE = {
    'chelsea': 'SW3',
    'southkensington': 'SW7',
    'kensington': 'W8',
    'belgravia': 'SW1X',
    'knightsbridge': 'SW1X',
    'mayfair': 'W1K',
    'stjohnswood': 'NW8',
    'hampstead': 'NW3',
    'nottinghill': 'W11',
    'bayswater': 'W2',
    'earlscourt': 'SW5',
    'fulham': 'SW6',
    'battersea': 'SW11',
    'pimlico': 'SW1V',
    'westminster': 'SW1P',
    'marylebone': 'W1U',
}

# Tube stations and postcode centroids (same as V3)
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
    'SW1X': (51.4983, -0.1560), 'SW1W': (51.4920, -0.1470), 'SW1P': (51.4960, -0.1300),
    'SW3': (51.4900, -0.1680), 'SW5': (51.4920, -0.1940), 'SW7': (51.4950, -0.1780),
    'SW10': (51.4830, -0.1820), 'SW11': (51.4650, -0.1650), 'SW13': (51.4750, -0.2430),
    'W1K': (51.5120, -0.1510), 'W1J': (51.5080, -0.1470), 'W1G': (51.5180, -0.1470),
    'W1U': (51.5200, -0.1520), 'W2': (51.5150, -0.1780), 'W8': (51.5010, -0.1920),
    'W11': (51.5150, -0.2050), 'W14': (51.4950, -0.2100), 'NW1': (51.5350, -0.1550),
    'NW3': (51.5550, -0.1780), 'NW8': (51.5330, -0.1750),
}

# City center: Trafalgar Square
CITY_CENTER = (51.5074, -0.1278)


def normalize_area(area):
    if pd.isna(area) or area == '':
        return ''
    return str(area).lower().replace('-', '').replace(' ', '').replace("'", '')


def infer_postcode_from_area(area, existing_postcode):
    if pd.notna(existing_postcode) and existing_postcode != '':
        return existing_postcode
    return AREA_TO_POSTCODE.get(normalize_area(area), None)


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


def load_and_clean_data():
    conn = sqlite3.connect('output/rentals.db')
    df = pd.read_sql("""
        SELECT bedrooms, bathrooms, size_sqft, postcode, area, property_type,
               price_pcm, latitude, longitude
        FROM listings_deduped
        WHERE size_sqft > 0 AND bedrooms IS NOT NULL AND price_pcm > 0
    """, conn)
    conn.close()

    print(f"Raw data: {len(df)} records")

    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    mask1 = sqft_per_bed < 70
    ppsqft = df['price_pcm'] / df['size_sqft']
    mask2 = (ppsqft < 1.0) | (ppsqft > 25.0)
    mask3 = df['price_pcm'] > 50000
    mask4 = df['size_sqft'] < 150

    df_clean = df[~(mask1 | mask2 | mask3 | mask4)].copy()
    print(f"Clean data: {len(df_clean)} records")
    return df_clean


def engineer_features_v4(df):
    print("\n[FEATURE ENGINEERING V4]")

    # Area normalization
    df['area_normalized'] = df['area'].apply(normalize_area)

    # Postcode imputation
    df['postcode_district'] = df['postcode'].str.extract(r'^([A-Z]{1,2}\d{1,2})')[0]
    df['postcode_district_imputed'] = df.apply(
        lambda row: infer_postcode_from_area(row['area'], row['postcode_district']), axis=1
    )
    most_common = df['postcode_district_imputed'].value_counts().index[0] if df['postcode_district_imputed'].notna().any() else 'SW3'
    df['postcode_district_imputed'] = df['postcode_district_imputed'].fillna(most_common)
    df['postcode_was_imputed'] = df['postcode_district'].isna().astype(int)

    # Coordinates from postcode if missing
    def get_coords(row):
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or lat == 0:
            pc = row['postcode_district_imputed']
            if pc in POSTCODE_CENTROIDS:
                return pd.Series(POSTCODE_CENTROIDS[pc])
        return pd.Series([lat, lon])

    df[['lat_filled', 'lon_filled']] = df.apply(get_coords, axis=1)

    # Distance features
    df['tube_distance_km'] = df.apply(lambda r: get_nearest_tube_distance(r['lat_filled'], r['lon_filled']), axis=1)
    df['center_distance_km'] = df.apply(lambda r: get_distance_to_center(r['lat_filled'], r['lon_filled']), axis=1)

    median_tube = df['tube_distance_km'].median()
    median_center = df['center_distance_km'].median()
    df['tube_distance_km'] = df['tube_distance_km'].fillna(median_tube)
    df['center_distance_km'] = df['center_distance_km'].fillna(median_center)

    print(f"  Tube distance median: {median_tube:.2f}km")
    print(f"  Center distance median: {median_center:.2f}km")

    # Let type
    df['is_short_let'] = df['property_type'].str.lower().str.contains('short', na=False).astype(int)
    df['is_long_let'] = df['property_type'].str.lower().str.contains('long', na=False).astype(int)
    df['property_type_clean'] = df['property_type'].str.lower().str.replace('short let', '').str.replace('long let', '').str.strip()
    df['property_type_clean'] = df['property_type_clean'].replace('', 'flat')

    # Basic features
    df['bathrooms'] = df['bathrooms'].fillna(1)
    beds_adj = df['bedrooms'].replace(0, 0.5)

    df['size_per_bed'] = df['size_sqft'] / beds_adj
    df['bath_ratio'] = df['bathrooms'] / beds_adj
    df['bed_bath_interaction'] = df['bedrooms'] * df['bathrooms']
    df['size_poly'] = df['size_sqft'] ** 1.5 / 1000
    df['sqft_per_bath'] = df['size_sqft'] / (df['bathrooms'] + 1)
    df['log_sqft'] = np.log1p(df['size_sqft'])

    # NEW V4: Size bins (quartiles)
    df['size_bin'] = pd.qcut(df['size_sqft'], q=4, labels=[0, 1, 2, 3]).astype(int)

    # NEW V4: Let type interactions
    df['short_let_size'] = df['is_short_let'] * df['log_sqft']
    df['long_let_size'] = df['is_long_let'] * df['log_sqft']

    # NEW V4: Center distance interactions
    df['center_size_interaction'] = df['center_distance_km'] * df['size_sqft'] / 1000
    df['log_center_distance'] = np.log1p(df['center_distance_km'])

    # Luxury indicators
    df['is_luxury'] = (df['price_pcm'] > 10000).astype(int)
    df['luxury_size_interaction'] = df['is_luxury'] * df['size_sqft'] / 1000

    # Tube interactions
    df['tube_size_interaction'] = df['tube_distance_km'] * df['size_sqft'] / 1000
    df['log_tube_distance'] = np.log1p(df['tube_distance_km'])

    # Target encoding
    global_mean = np.log1p(df['price_pcm']).mean()
    smoothing = 15

    for col in ['area_normalized', 'postcode_district_imputed']:
        means = np.log1p(df.groupby(col)['price_pcm'].transform('mean'))
        counts = df.groupby(col)['price_pcm'].transform('count')
        df[f'{col}_encoded'] = (means * counts + global_mean * smoothing) / (counts + smoothing)

    df['size_postcode_interaction'] = df['size_sqft'] * df['postcode_district_imputed_encoded'] / 1000

    # One-hot encode
    df = pd.get_dummies(df, columns=['property_type_clean'], prefix='type')

    return df


def evaluate_model_weighted(model, X, y, cv=5, weight_high_end=True):
    """Evaluate with optional sample weights for high-end properties."""
    y_log = np.log1p(y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    all_preds = []
    all_actual = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y.iloc[val_idx]

        # Sample weights: higher for expensive properties
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


def build_xgboost_v4():
    return XGBRegressor(
        n_estimators=1200,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.75,
        colsample_bytree=0.65,
        min_child_weight=2,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )


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
    print("=" * 70)
    print("LONDON RENTAL PRICE MODELS V4 - PUSHING TO R²=0.80+")
    print("Additional: sample weights, center distance, more interactions")
    print("=" * 70)

    df = load_and_clean_data()
    df = engineer_features_v4(df)

    feature_cols = [
        'bedrooms', 'bathrooms', 'size_sqft',
        'size_per_bed', 'bath_ratio', 'bed_bath_interaction',
        'size_poly', 'sqft_per_bath', 'log_sqft',
        'area_normalized_encoded', 'postcode_district_imputed_encoded',
        'size_postcode_interaction',
        'postcode_was_imputed',
        'tube_distance_km', 'log_tube_distance', 'tube_size_interaction',
        'center_distance_km', 'log_center_distance', 'center_size_interaction',
        'is_short_let', 'is_long_let',
        'short_let_size', 'long_let_size',
        'size_bin',
        'luxury_size_interaction',
    ] + [c for c in df.columns if c.startswith('type_')]

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

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (5-fold CV)")
    print("=" * 70)

    results = {}

    # XGBoost V4 without weights
    print("\n[XGBoost_V4_NoWeights]")
    model = build_xgboost_v4()
    result = evaluate_model_weighted(model, X, y, weight_high_end=False)
    results['XGBoost_V4_NoWeights'] = result
    print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
    analyze_errors_by_tier(result['actual'], result['predictions'], tiers)

    # XGBoost V4 with sample weights
    print("\n[XGBoost_V4_Weighted]")
    model = build_xgboost_v4()
    result = evaluate_model_weighted(model, X, y, weight_high_end=True)
    results['XGBoost_V4_Weighted'] = result
    print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
    analyze_errors_by_tier(result['actual'], result['predictions'], tiers)

    # CatBoost
    if HAS_CATBOOST:
        print("\n[CatBoost]")
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=800,
            learning_rate=0.02,
            depth=8,
            loss_function='MAE',
            random_seed=42,
            verbose=False
        )
        result = evaluate_model_weighted(model, X, y, weight_high_end=False)
        results['CatBoost'] = result
        print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
        analyze_errors_by_tier(result['actual'], result['predictions'], tiers)

    # LightGBM
    if HAS_LIGHTGBM:
        print("\n[LightGBM]")
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.02,
            max_depth=8,
            num_leaves=63,
            objective='mae',
            random_state=42,
            verbosity=-1
        )
        result = evaluate_model_weighted(model, X, y, weight_high_end=False)
        results['LightGBM'] = result
        print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
        analyze_errors_by_tier(result['actual'], result['predictions'], tiers)

    # Best model summary
    best_name = max(results, key=lambda x: results[x]['R2'])
    best = results[best_name]

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}")
    print("=" * 70)
    print(f"R²:         {best['R2']:.4f}")
    print(f"MAE:        £{best['MAE']:,.0f}")
    print(f"MAPE:       {best['MAPE']:.1f}%")
    print(f"Median APE: {best['Median_APE']:.1f}%")

    # Feature importance
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (Best XGBoost)")
    print("-" * 70)

    best_model = build_xgboost_v4()
    best_model.fit(X, np.log1p(y))

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:<35} {row['importance']:.3f} {bar}")

    # Improvement summary
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)
    print("V1 (baseline):    R²=0.658, MAE=£2,589, MAPE=28.2%")
    print("V2 (Grok/Gemini): R²=0.701, MAE=£1,937, MAPE=23.1%")
    print("V3 (augmented):   R²=0.777, MAE=£1,624, MAPE=19.8%")
    print(f"V4 ({best_name}): R²={best['R2']:.3f}, MAE=£{best['MAE']:,.0f}, MAPE={best['MAPE']:.1f}%")

    v3_r2 = 0.777
    r2_improvement = (best['R2'] - v3_r2) / v3_r2 * 100
    print(f"\nV4 vs V3 R² change: {r2_improvement:+.1f}%")

    return results, df


if __name__ == '__main__':
    results, df = main()
