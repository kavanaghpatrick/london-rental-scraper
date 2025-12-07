"""
London Rental Price Prediction Models V3 - Data Augmentation

Improvements based on Grok analysis:
1. Area name normalization (case, hyphens, variants)
2. Postcode imputation from area names
3. Prime London area-to-postcode mapping
4. Tube station distance features
5. Let type separation (long_let vs short_let)
6. Luxury tier interaction features
7. Stacking ensemble

Usage:
    python rental_price_models_v3.py
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


# Prime London area to postcode district mapping
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
    'fitzrovia': 'W1T',
    'hollandpark': 'W11',
    'canarywharf': 'E14',
    'regentspark': 'NW1',
    'primrosehill': 'NW1',
}

# Tube station coordinates for Prime London (manually curated key stations)
TUBE_STATIONS = {
    'South Kensington': (51.4941, -0.1738),
    'Sloane Square': (51.4924, -0.1565),
    'Knightsbridge': (51.5015, -0.1607),
    'Hyde Park Corner': (51.5027, -0.1527),
    'Green Park': (51.5067, -0.1428),
    'Bond Street': (51.5142, -0.1494),
    'Marble Arch': (51.5136, -0.1586),
    'Notting Hill Gate': (51.5094, -0.1967),
    'Holland Park': (51.5075, -0.2060),
    'High Street Kensington': (51.5009, -0.1925),
    'Earls Court': (51.4914, -0.1934),
    'Gloucester Road': (51.4945, -0.1829),
    'St Johns Wood': (51.5347, -0.1740),
    'Swiss Cottage': (51.5432, -0.1747),
    'Hampstead': (51.5566, -0.1780),
    'Baker Street': (51.5226, -0.1571),
    'Regents Park': (51.5234, -0.1466),
    'Paddington': (51.5154, -0.1755),
    'Victoria': (51.4965, -0.1447),
    'Pimlico': (51.4893, -0.1334),
    'Westminster': (51.5014, -0.1248),
    'Fulham Broadway': (51.4802, -0.1953),
    'Parsons Green': (51.4753, -0.2010),
    'Queensway': (51.5107, -0.1871),
    'Lancaster Gate': (51.5119, -0.1756),
    'Canary Wharf': (51.5033, -0.0184),
    'Clapham Junction': (51.4652, -0.1703),
}

# Postcode district centroids (approximate)
POSTCODE_CENTROIDS = {
    'SW1X': (51.4983, -0.1560),  # Belgravia
    'SW1W': (51.4920, -0.1470),  # Pimlico
    'SW1P': (51.4960, -0.1300),  # Westminster
    'SW1V': (51.4890, -0.1370),  # Pimlico
    'SW3': (51.4900, -0.1680),   # Chelsea
    'SW5': (51.4920, -0.1940),   # Earls Court
    'SW7': (51.4950, -0.1780),   # South Kensington
    'SW10': (51.4830, -0.1820),  # West Brompton
    'SW11': (51.4650, -0.1650),  # Battersea
    'SW13': (51.4750, -0.2430),  # Barnes
    'SW14': (51.4650, -0.2650),  # Mortlake
    'SW18': (51.4550, -0.1880),  # Wandsworth
    'SW19': (51.4220, -0.2060),  # Wimbledon
    'W1K': (51.5120, -0.1510),   # Mayfair
    'W1J': (51.5080, -0.1470),   # Mayfair
    'W1G': (51.5180, -0.1470),   # Marylebone
    'W1H': (51.5170, -0.1590),   # Marylebone
    'W1U': (51.5200, -0.1520),   # Marylebone
    'W2': (51.5150, -0.1780),    # Bayswater
    'W8': (51.5010, -0.1920),    # Kensington
    'W11': (51.5150, -0.2050),   # Notting Hill
    'W14': (51.4950, -0.2100),   # West Kensington
    'NW1': (51.5350, -0.1550),   # Regents Park
    'NW3': (51.5550, -0.1780),   # Hampstead
    'NW8': (51.5330, -0.1750),   # St Johns Wood
}


def normalize_area(area):
    """Normalize area name for consistent matching."""
    if pd.isna(area) or area == '':
        return ''
    # Lowercase, remove hyphens and spaces
    clean = str(area).lower().replace('-', '').replace(' ', '').replace("'", '')
    return clean


def infer_postcode_from_area(area, existing_postcode):
    """Infer postcode district from area name if missing."""
    if pd.notna(existing_postcode) and existing_postcode != '':
        return existing_postcode

    norm_area = normalize_area(area)
    return AREA_TO_POSTCODE.get(norm_area, None)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def get_nearest_tube_distance(lat, lon):
    """Get distance to nearest tube station in km."""
    if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
        return None

    min_dist = float('inf')
    for station, (slat, slon) in TUBE_STATIONS.items():
        dist = haversine_distance(lat, lon, slat, slon)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def get_tube_distance_from_postcode(postcode_district):
    """Get estimated tube distance from postcode centroid."""
    if postcode_district not in POSTCODE_CENTROIDS:
        return None

    lat, lon = POSTCODE_CENTROIDS[postcode_district]
    return get_nearest_tube_distance(lat, lon)


def load_and_clean_data():
    """Load data and apply all cleaning rules."""
    conn = sqlite3.connect('output/rentals.db')
    df = pd.read_sql("""
        SELECT bedrooms, bathrooms, size_sqft, postcode, area, property_type,
               price_pcm, latitude, longitude, furnished
        FROM listings_deduped
        WHERE size_sqft > 0 AND bedrooms IS NOT NULL AND price_pcm > 0
    """, conn)
    conn.close()

    print(f"Raw data: {len(df)} records")

    # CLEANING RULE 1: Remove impossible sqft/bedroom ratios
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    mask_density = sqft_per_bed < 70
    print(f"  Removed {mask_density.sum()} with <70 sqft/bedroom")

    # CLEANING RULE 2: Remove price/sqft outliers
    ppsqft = df['price_pcm'] / df['size_sqft']
    mask_ppsqft = (ppsqft < 1.0) | (ppsqft > 25.0)
    print(f"  Removed {mask_ppsqft.sum()} with price/sqft outliers")

    # CLEANING RULE 3: Cap extreme prices
    mask_cap = df['price_pcm'] > 50000
    print(f"  Removed {mask_cap.sum()} with price >£50k")

    # CLEANING RULE 4: Remove extremely small properties
    mask_small = df['size_sqft'] < 150
    print(f"  Removed {mask_small.sum()} with size <150 sqft")

    # Apply all filters
    df_clean = df[~(mask_density | mask_ppsqft | mask_cap | mask_small)].copy()
    print(f"Clean data: {len(df_clean)} records ({100*len(df_clean)/len(df):.1f}% retained)")

    return df_clean


def engineer_features_v3(df):
    """Apply V3 feature engineering with area normalization and external data."""

    print("\n[FEATURE ENGINEERING V3]")

    # ===== AREA NORMALIZATION =====
    df['area_normalized'] = df['area'].apply(normalize_area)
    unique_before = df['area'].nunique()
    unique_after = df['area_normalized'].nunique()
    print(f"  Area normalization: {unique_before} -> {unique_after} unique values")

    # ===== POSTCODE IMPUTATION =====
    missing_postcode_before = df['postcode'].isna().sum() + (df['postcode'] == '').sum()

    # Extract postcode district
    df['postcode_district'] = df['postcode'].str.extract(r'^([A-Z]{1,2}\d{1,2})')[0]

    # Impute from area where missing
    df['postcode_district_imputed'] = df.apply(
        lambda row: infer_postcode_from_area(row['area'], row['postcode_district']),
        axis=1
    )

    # Fill remaining with most common
    most_common = df['postcode_district_imputed'].value_counts().index[0] if df['postcode_district_imputed'].notna().any() else 'SW3'
    df['postcode_district_imputed'] = df['postcode_district_imputed'].fillna(most_common)

    # Add imputation flag
    df['postcode_was_imputed'] = (df['postcode_district'].isna() | (df['postcode_district'] == '')).astype(int)

    missing_after = df['postcode_district_imputed'].isna().sum()
    print(f"  Postcode imputation: {missing_postcode_before} missing -> {missing_after} missing")

    # ===== TUBE DISTANCE =====
    # Try lat/lng first, fallback to postcode centroid
    def get_tube_dist(row):
        dist = get_nearest_tube_distance(row['latitude'], row['longitude'])
        if dist is None:
            dist = get_tube_distance_from_postcode(row['postcode_district_imputed'])
        return dist

    df['tube_distance_km'] = df.apply(get_tube_dist, axis=1)
    # Fill remaining with median
    median_tube = df['tube_distance_km'].median()
    df['tube_distance_km'] = df['tube_distance_km'].fillna(median_tube)
    print(f"  Tube distance: median={median_tube:.2f}km")

    # ===== LET TYPE EXTRACTION =====
    df['is_short_let'] = df['property_type'].str.lower().str.contains('short', na=False).astype(int)
    df['is_long_let'] = df['property_type'].str.lower().str.contains('long', na=False).astype(int)

    # Clean property type (remove let type from it)
    df['property_type_clean'] = df['property_type'].str.lower().str.replace('short let', '').str.replace('long let', '').str.strip()
    df['property_type_clean'] = df['property_type_clean'].replace('', 'flat')

    print(f"  Let types: short_let={df['is_short_let'].sum()}, long_let={df['is_long_let'].sum()}")

    # ===== BASIC FEATURES (from V2) =====
    df['bathrooms'] = df['bathrooms'].fillna(1)
    beds_adj = df['bedrooms'].replace(0, 0.5)

    df['size_per_bed'] = df['size_sqft'] / beds_adj
    df['bath_ratio'] = df['bathrooms'] / beds_adj
    df['bed_bath_interaction'] = df['bedrooms'] * df['bathrooms']
    df['size_poly'] = df['size_sqft'] ** 1.5 / 1000
    df['sqft_per_bath'] = df['size_sqft'] / (df['bathrooms'] + 1)
    df['log_sqft'] = np.log1p(df['size_sqft'])

    # ===== LUXURY TIER FEATURES =====
    # Create luxury indicator
    df['is_luxury'] = (df['price_pcm'] > 10000).astype(int)
    df['is_ultra_luxury'] = (df['price_pcm'] > 20000).astype(int)

    # Luxury interaction with size
    df['luxury_size_interaction'] = df['is_luxury'] * df['size_sqft'] / 1000

    # ===== TUBE DISTANCE INTERACTIONS =====
    df['tube_size_interaction'] = df['tube_distance_km'] * df['size_sqft'] / 1000
    df['log_tube_distance'] = np.log1p(df['tube_distance_km'])

    # ===== TARGET ENCODING =====
    global_mean = np.log1p(df['price_pcm']).mean()
    smoothing = 15

    for col in ['area_normalized', 'postcode_district_imputed']:
        means = np.log1p(df.groupby(col)['price_pcm'].transform('mean'))
        counts = df.groupby(col)['price_pcm'].transform('count')
        df[f'{col}_encoded'] = (means * counts + global_mean * smoothing) / (counts + smoothing)

    # Size-postcode interaction
    df['size_postcode_interaction'] = df['size_sqft'] * df['postcode_district_imputed_encoded'] / 1000

    # ===== ONE-HOT ENCODE PROPERTY TYPE =====
    df = pd.get_dummies(df, columns=['property_type_clean'], prefix='type')

    return df


def build_stacking_ensemble():
    """Build stacking ensemble as recommended by Grok."""
    base_models = [
        ('xgb', XGBRegressor(
            n_estimators=800,
            learning_rate=0.015,
            max_depth=6,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='reg:absoluteerror',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )),
        ('ridge', Ridge(alpha=1.0))
    ]

    meta_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        verbosity=0
    )

    return StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )


def build_xgboost_v3():
    """Build XGBoost with V3 improvements."""
    return XGBRegressor(
        n_estimators=1000,
        learning_rate=0.012,
        max_depth=7,
        subsample=0.75,
        colsample_bytree=0.70,
        min_child_weight=2,
        reg_alpha=0.15,
        reg_lambda=1.5,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )


def build_quantile_model(quantile):
    """Build quantile regression model."""
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.7,
        objective='reg:quantileerror',
        quantile_alpha=quantile,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )


def evaluate_model(model, X, y, cv=5, model_name="Model"):
    """Evaluate model with detailed metrics."""
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

    mae = mean_absolute_error(all_actual, all_preds)
    rmse = np.sqrt(mean_squared_error(all_actual, all_preds))
    r2 = r2_score(all_actual, all_preds)
    mape = np.mean(np.abs((all_actual - all_preds) / all_actual)) * 100
    median_ape = np.median(np.abs((all_actual - all_preds) / all_actual)) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Median_APE': median_ape,
        'predictions': all_preds,
        'actual': all_actual
    }


def analyze_errors_by_tier(actual, preds, tiers):
    """Analyze prediction errors by price tier."""
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
    print("LONDON RENTAL PRICE MODELS V3 - DATA AUGMENTATION")
    print("Based on Grok recommendations: area normalization, postcode imputation,")
    print("tube distances, let type separation, luxury interactions, stacking")
    print("=" * 70)

    # Load and clean data
    df = load_and_clean_data()

    # Engineer features
    df = engineer_features_v3(df)

    # Define feature columns
    feature_cols = [
        # Core features
        'bedrooms', 'bathrooms', 'size_sqft',
        # Ratio features
        'size_per_bed', 'bath_ratio', 'bed_bath_interaction',
        # Polynomial/log features
        'size_poly', 'sqft_per_bath', 'log_sqft',
        # Location encoded
        'area_normalized_encoded', 'postcode_district_imputed_encoded',
        'size_postcode_interaction',
        # NEW: Postcode imputation flag
        'postcode_was_imputed',
        # NEW: Tube distance features
        'tube_distance_km', 'log_tube_distance', 'tube_size_interaction',
        # NEW: Let type features
        'is_short_let', 'is_long_let',
        # NEW: Luxury interactions
        'luxury_size_interaction',
    ] + [c for c in df.columns if c.startswith('type_')]

    X = df[feature_cols].fillna(0)
    y = df['price_pcm']

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target: £{y.min():,.0f} - £{y.max():,.0f} (mean £{y.mean():,.0f})")

    # Price tiers
    tiers = [
        (0, 3000, '<£3k'),
        (3000, 5000, '£3-5k'),
        (5000, 10000, '£5-10k'),
        (10000, 20000, '£10-20k'),
        (20000, 100000, '£20k+')
    ]

    # Evaluate models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (5-fold CV)")
    print("=" * 70)

    models = {
        'XGBoost_V3': build_xgboost_v3(),
        'Stacking_Ensemble': build_stacking_ensemble(),
    }

    results = {}
    for name, model in models.items():
        print(f"\n[{name}]")
        result = evaluate_model(model, X, y, model_name=name)
        results[name] = result

        print(f"  R²:         {result['R2']:.4f}")
        print(f"  MAE:        £{result['MAE']:,.0f}")
        print(f"  RMSE:       £{result['RMSE']:,.0f}")
        print(f"  MAPE:       {result['MAPE']:.1f}%")
        print(f"  Median APE: {result['Median_APE']:.1f}%")

        analyze_errors_by_tier(result['actual'], result['predictions'], tiers)

    # Best model
    best_name = min(results, key=lambda x: results[x]['MAE'])
    best = results[best_name]

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}")
    print("=" * 70)
    print(f"R²:         {best['R2']:.4f}")
    print(f"MAE:        £{best['MAE']:,.0f}")
    print(f"MAPE:       {best['MAPE']:.1f}%")
    print(f"Median APE: {best['Median_APE']:.1f}%")

    # Quantile models for prediction intervals
    print("\n" + "-" * 70)
    print("QUANTILE REGRESSION (Prediction Intervals)")
    print("-" * 70)

    y_log = np.log1p(y)
    quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        model = build_quantile_model(q)
        model.fit(X, y_log)
        quantile_models[q] = model
        print(f"  Trained q={q:.1f} model")

    # Feature importance
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (Best Model)")
    print("-" * 70)

    # Train best model on full data for feature importance
    best_model = build_xgboost_v3()
    best_model.fit(X, np.log1p(y))

    if hasattr(best_model, 'feature_importances_'):
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
    print("V1 (XGBoost baseline):  R²=0.658, MAE=£2,589, MAPE=28.2%")
    print("V2 (Grok/Gemini):       R²=0.701, MAE=£1,937, MAPE=23.1%")
    print(f"V3 ({best_name}):       R²={best['R2']:.3f}, MAE=£{best['MAE']:,.0f}, MAPE={best['MAPE']:.1f}%")

    v2_r2 = 0.701
    v2_mae = 1937
    v2_mape = 23.1

    r2_improvement = (best['R2'] - v2_r2) / v2_r2 * 100
    mae_improvement = (v2_mae - best['MAE']) / v2_mae * 100
    mape_improvement = (v2_mape - best['MAPE']) / v2_mape * 100

    print(f"\nV3 vs V2 Improvements:")
    print(f"  R² improved by:   {r2_improvement:+.1f}%")
    print(f"  MAE improved by:  {mae_improvement:+.1f}%")
    print(f"  MAPE improved by: {mape_improvement:+.1f}%")

    return results, quantile_models, df


if __name__ == '__main__':
    results, quantile_models, df = main()
