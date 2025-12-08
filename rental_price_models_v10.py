"""
London Rental Price Prediction Models V10 - Enhanced Floor Features

Building on V9 (R²=0.839), adding comprehensive floor-level features:
1. property_levels encoding (single/duplex/triplex/multi_floor)
2. Floor flag features (basement, lower_ground, ground, etc.)
3. Vertical luxury score (weighted floor diversity)
4. Floor-location premium interactions
5. Basement premium (full house indicator)
6. Multi-floor centrality interactions

Data analysis shows strong correlations:
- 7-floor: £46,362/mo vs 1-floor: £5,685/mo (8x difference)
- Basement properties: £22,037 avg (premium full houses)
- Multi-floor: £19,537 vs Single-floor: £6,206 (3x difference)

Baseline (V9): R²=0.839, MAE=£1,560, MAPE=19.6%

Usage:
    python rental_price_models_v10.py
    python rental_price_models_v10.py --quick    # Skip Optuna tuning
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

AMENITY_WEIGHTS = {
    'has_pool': 3.0, 'has_gym': 2.0, 'has_porter': 2.5, 'has_ac': 1.5,
    'has_lift': 1.0, 'has_view': 2.0, 'has_high_ceilings': 1.5,
    'has_terrace': 1.5, 'has_roof_terrace': 2.0, 'has_garden': 1.5,
    'has_balcony': 1.0, 'has_parking': 1.5, 'has_furnished': 0.5,
    'has_modern': 0.5, 'has_period': 1.0,
}

# V10: Floor type weights for vertical luxury score
FLOOR_WEIGHTS = {
    'has_basement': 3.0,        # Full houses typically
    'has_lower_ground': 2.0,    # Extra living space
    'has_roof_terrace': 2.5,    # Premium outdoor
    'has_mezzanine': 1.5,       # Architectural interest
    'has_fourth_plus': 1.0,     # Upper floors
}

PREMIUM_AGENTS = {'savills', 'knightfrank', 'chestertons', 'hamptons', 'dexters'}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def parse_amenities(features_str):
    result = {col: 0 for col in AMENITY_FEATURES}
    if not features_str or pd.isna(features_str):
        return result
    text = str(features_str).lower()
    if 'balcony' in text: result['has_balcony'] = 1
    if 'terrace' in text and 'roof' not in text: result['has_terrace'] = 1
    if 'garden' in text: result['has_garden'] = 1
    if 'concierge' in text or 'porter' in text or '24 hour' in text or '24-hour' in text: result['has_porter'] = 1
    if 'gym' in text or 'fitness' in text: result['has_gym'] = 1
    if 'pool' in text or 'swimming' in text: result['has_pool'] = 1
    if 'parking' in text or 'garage' in text: result['has_parking'] = 1
    if 'lift' in text or 'elevator' in text: result['has_lift'] = 1
    if 'air con' in text or 'air-con' in text or 'a/c' in text or 'cooling' in text: result['has_ac'] = 1
    if 'furnished' in text and 'unfurnished' not in text: result['has_furnished'] = 1
    if 'high ceiling' in text or 'tall ceiling' in text or 'double height' in text: result['has_high_ceilings'] = 1
    if 'view' in text and ('park' in text or 'river' in text or 'garden' in text or 'city' in text): result['has_view'] = 1
    if 'modern' in text or 'contemporary' in text or 'new build' in text: result['has_modern'] = 1
    if 'period' in text or 'victorian' in text or 'georgian' in text or 'edwardian' in text: result['has_period'] = 1
    return result


def parse_room_details(room_details_str):
    result = {
        'room_diversity': 0, 'non_bedroom_count': 0, 'total_rooms': 0,
        'has_reception': 0, 'has_kitchen': 0, 'has_dining': 0,
        'has_study': 0, 'has_utility': 0
    }
    if not room_details_str or pd.isna(room_details_str):
        return result
    try:
        rooms = json.loads(room_details_str)
        if isinstance(rooms, list) and len(rooms) > 0:
            room_types = set()
            non_bedroom = 0
            for room in rooms:
                if isinstance(room, dict):
                    room_type = room.get('type', '').lower()
                    room_name = room.get('name', '').lower()
                    room_types.add(room_type)
                    if 'bedroom' not in room_type and 'bedroom' not in room_name:
                        non_bedroom += 1
                    if 'reception' in room_type or 'living' in room_name or 'lounge' in room_name:
                        result['has_reception'] = 1
                    if 'kitchen' in room_type or 'kitchen' in room_name:
                        result['has_kitchen'] = 1
                    if 'dining' in room_type or 'dining' in room_name:
                        result['has_dining'] = 1
                    if 'study' in room_type or 'study' in room_name or 'office' in room_name:
                        result['has_study'] = 1
                    if 'utility' in room_type or 'utility' in room_name:
                        result['has_utility'] = 1
            result['room_diversity'] = len(room_types)
            result['non_bedroom_count'] = non_bedroom
            result['total_rooms'] = len(rooms)
    except (json.JSONDecodeError, TypeError):
        pass
    return result


def load_and_clean_data():
    """Load data with enhanced floor columns."""
    conn = sqlite3.connect('output/rentals.db')
    df = pd.read_sql("""
        SELECT
            bedrooms, bathrooms, size_sqft,
            postcode, area, property_type,
            price_pcm, latitude, longitude, features, source,
            property_type_std, let_type, postcode_normalized,
            postcode_inferred, agent_brand, room_details,
            floor_count, has_roof_terrace,
            property_levels, has_basement, has_lower_ground,
            has_ground, has_mezzanine, has_first_floor,
            has_second_floor, has_third_floor, has_fourth_plus
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

    # V10: Report floor data coverage
    floor_count_coverage = (df_clean['floor_count'] > 0).sum() / len(df_clean) * 100
    levels_coverage = df_clean['property_levels'].notna().sum() / len(df_clean) * 100
    print(f"Floor count coverage: {floor_count_coverage:.1f}%")
    print(f"Property levels coverage: {levels_coverage:.1f}%")

    return df_clean


def engineer_features_v10(df):
    """V10 feature engineering with enhanced floor features."""
    print("\n[FEATURE ENGINEERING V10 - FLOOR FEATURES]")

    # ========== AMENITY FEATURES ==========
    amenity_dicts = df['features'].apply(parse_amenities)
    amenity_df = pd.DataFrame(amenity_dicts.tolist())
    for col in AMENITY_FEATURES:
        df[col] = amenity_df[col].values

    df['amenity_score'] = df[AMENITY_FEATURES].sum(axis=1)
    luxury_amenities = ['has_pool', 'has_porter', 'has_gym', 'has_ac']
    df['luxury_amenity_score'] = df[luxury_amenities].sum(axis=1)
    df['has_outdoor_space'] = ((df['has_balcony'] == 1) | (df['has_terrace'] == 1) | (df['has_garden'] == 1)).astype(int)
    df['amenity_density'] = df['amenity_score'] / df['size_sqft'] * 1000

    df['weighted_luxury_score'] = sum(
        df.get(amenity, 0) * weight
        for amenity, weight in AMENITY_WEIGHTS.items()
        if amenity in df.columns
    )

    # ========== ROOM DETAILS ==========
    room_dicts = df['room_details'].apply(parse_room_details)
    room_df = pd.DataFrame(room_dicts.tolist())
    for col in room_df.columns:
        df[col] = room_df[col].values

    # ========== PROPERTY TYPE ==========
    df['property_type_std'] = df['property_type_std'].fillna('flat')
    type_encoding = {'studio': 0, 'flat': 1, 'maisonette': 2, 'house': 3, 'penthouse': 4, 'other': 1, 'serviced': 1}
    df['property_type_encoded'] = df['property_type_std'].map(type_encoding).fillna(1)
    df['prop_type_size'] = df['property_type_encoded'] * df['size_sqft'] / 1000

    # ========== LET TYPE ==========
    df['let_type'] = df['let_type'].fillna('unknown')
    df['is_short_let'] = (df['let_type'] == 'short').astype(int)
    df['is_long_let'] = (df['let_type'] == 'long').astype(int)

    # ========== POSTCODE ==========
    df['postcode_district'] = df['postcode_normalized'].fillna('SW3')
    df['postcode_was_inferred'] = df['postcode_inferred'].fillna(0).astype(int)
    df['postcode_area'] = df['postcode_district'].str.extract(r'^([A-Z]+)', expand=False).fillna('SW')

    # ========== AGENT BRAND ==========
    df['agent_brand'] = df['agent_brand'].fillna('unknown')
    df['is_premium_agent'] = df['source'].apply(lambda x: 1 if str(x).lower() in PREMIUM_AGENTS else 0)
    agent_encoding = {'savills': 4, 'knightfrank': 4, 'chestertons': 3, 'foxtons': 3, 'rightmove': 1, 'johndwood': 2}
    df['agent_brand_encoded'] = df['source'].map(agent_encoding).fillna(1)

    # ========== NUMERIC FEATURES ==========
    df['bedrooms'] = df['bedrooms'].fillna(1)
    df['bathrooms'] = df['bathrooms'].fillna(1)
    df['log_sqft'] = np.log1p(df['size_sqft'])
    df['sqft_per_bed'] = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    df['bath_ratio'] = df['bathrooms'] / df['bedrooms'].replace(0, 0.5)
    df['beds_squared'] = df['bedrooms'] ** 2

    # ========== LOCATION FEATURES ==========
    df['latitude'] = df['latitude'].fillna(51.5074)
    df['longitude'] = df['longitude'].fillna(-0.1278)
    df['center_distance_km'] = haversine(df['latitude'], df['longitude'], CITY_CENTER[0], CITY_CENTER[1])
    df['center_distance_inverse'] = 1 / (df['center_distance_km'] + 0.5)
    df['center_distance_squared'] = df['center_distance_km'] ** 2
    df['center_distance_log'] = np.log1p(df['center_distance_km'])
    df['central_size_premium'] = np.exp(-df['center_distance_km'] / 3) * df['log_sqft']

    # Nearest tube
    min_tube = float('inf')
    for name, (lat, lon) in TUBE_STATIONS.items():
        dist = haversine(df['latitude'], df['longitude'], lat, lon)
        min_tube = np.minimum(min_tube, dist)
    df['tube_distance_km'] = min_tube

    # ========== V10: ENHANCED FLOOR FEATURES ==========
    print("\n  [V10 FLOOR FEATURES]")

    # Fill missing floor data
    df['floor_count'] = df['floor_count'].fillna(0)
    df['has_roof_terrace'] = df['has_roof_terrace'].fillna(0)
    df['property_levels'] = df['property_levels'].fillna('')

    # Fill floor flags
    floor_flags = ['has_basement', 'has_lower_ground', 'has_ground', 'has_mezzanine',
                   'has_first_floor', 'has_second_floor', 'has_third_floor', 'has_fourth_plus']
    for flag in floor_flags:
        df[flag] = df[flag].fillna(0)

    # 1. Property levels encoding (ordinal)
    levels_encoding = {'single_floor': 1, 'duplex': 2, 'triplex': 3, 'multi_floor': 4}
    df['property_levels_encoded'] = df['property_levels'].map(levels_encoding).fillna(0)
    print(f"    Property levels encoded: {(df['property_levels_encoded'] > 0).sum()} records")

    # 2. Binary floor type indicators
    df['is_single_floor'] = (df['property_levels'] == 'single_floor').astype(int)
    df['is_duplex'] = (df['property_levels'] == 'duplex').astype(int)
    df['is_triplex'] = (df['property_levels'] == 'triplex').astype(int)
    df['is_multi_floor'] = (df['property_levels'] == 'multi_floor').astype(int)

    # 3. Floor diversity (number of different floor types)
    df['floor_diversity'] = (
        df['has_basement'].astype(int) + df['has_lower_ground'].astype(int) +
        df['has_ground'].astype(int) + df['has_mezzanine'].astype(int) +
        df['has_first_floor'].astype(int) + df['has_second_floor'].astype(int) +
        df['has_third_floor'].astype(int) + df['has_fourth_plus'].astype(int)
    )
    print(f"    Floor diversity: mean={df[df['floor_diversity'] > 0]['floor_diversity'].mean():.2f}")

    # 4. Vertical luxury score (weighted floor features)
    df['vertical_luxury_score'] = (
        df['has_basement'] * FLOOR_WEIGHTS['has_basement'] +
        df['has_lower_ground'] * FLOOR_WEIGHTS['has_lower_ground'] +
        df['has_roof_terrace'] * FLOOR_WEIGHTS['has_roof_terrace'] +
        df['has_mezzanine'] * FLOOR_WEIGHTS['has_mezzanine'] +
        df['has_fourth_plus'] * FLOOR_WEIGHTS['has_fourth_plus']
    )
    print(f"    Vertical luxury score: mean={df[df['vertical_luxury_score'] > 0]['vertical_luxury_score'].mean():.2f}")

    # 5. Floor-size interactions
    df['floor_span_size'] = df['floor_count'] * df['size_sqft'] / 1000
    df['floor_diversity_size'] = df['floor_diversity'] * df['size_sqft'] / 1000
    df['levels_size_interaction'] = df['property_levels_encoded'] * df['size_sqft'] / 1000

    # 6. Basement premium features (basements typically indicate full houses)
    df['basement_premium'] = df['has_basement'] * df['size_sqft'] / 1000
    df['basement_beds'] = df['has_basement'] * df['bedrooms']
    print(f"    Basement properties: {df['has_basement'].sum()}")

    # 7. Multi-floor premium features
    df['multi_floor_beds'] = df['is_multi_floor'] * df['bedrooms']
    df['duplex_size'] = df['is_duplex'] * df['size_sqft'] / 1000
    df['triplex_size'] = df['is_triplex'] * df['size_sqft'] / 1000

    # 8. Roof terrace premium (luxury indicator)
    df['roof_terrace_size'] = df['has_roof_terrace'] * df['size_sqft'] / 1000

    # ========== INTERACTIONS (from V9) ==========
    df['short_let_size'] = df['is_short_let'] * df['log_sqft']
    df['long_let_size'] = df['is_long_let'] * df['log_sqft']
    df['short_let_centrality'] = df['is_short_let'] * df['center_distance_inverse']
    df['center_size_interaction'] = df['center_distance_km'] * df['size_sqft'] / 1000
    df['tube_size_interaction'] = df['tube_distance_km'] * df['size_sqft'] / 1000
    df['amenity_size_interaction'] = df['amenity_score'] * df['log_sqft']
    df['outdoor_size_interaction'] = df['has_outdoor_space'] * df['size_sqft'] / 1000
    df['luxury_center_interaction'] = df['luxury_amenity_score'] * (5 - df['center_distance_km']).clip(lower=0)
    df['premium_agent_size'] = df['is_premium_agent'] * df['log_sqft']
    df['beds_center_interaction'] = df['bedrooms'] * (5 - df['center_distance_km']).clip(lower=0)
    df['premium_center_interaction'] = df['is_premium_agent'] * (5 - df['center_distance_km']).clip(lower=0)
    df['short_let_premium'] = df['is_short_let'] * df['is_premium_agent']
    df['amenity_beds_interaction'] = df['amenity_score'] * df['bedrooms']

    # ========== LUXURY INDICATORS ==========
    df['is_luxury'] = (df['price_pcm'] > 10000).astype(int)
    df['luxury_size_interaction'] = df['is_luxury'] * df['size_sqft'] / 1000
    df['luxury_bed_interaction'] = df['is_luxury'] * df['bedrooms']
    df['luxury_bath_interaction'] = df['is_luxury'] * df['bathrooms']
    df['premium_agent_luxury'] = df['is_premium_agent'] * df['is_luxury']
    df['bath_bed_ratio_luxury'] = df['bath_ratio'] * df['is_luxury']
    df['bath_bed_ratio_standard'] = df['bath_ratio'] * (1 - df['is_luxury'])

    # V10: Floor-luxury interactions
    df['luxury_floor_count'] = df['is_luxury'] * df['floor_count']
    df['luxury_vertical'] = df['is_luxury'] * df['vertical_luxury_score']
    print(f"    Luxury floor count avg: {df[df['is_luxury']==1]['floor_count'].mean():.2f}")

    # ========== TARGET ENCODING ==========
    global_mean = np.log1p(df['price_pcm']).mean()
    smoothing = 15

    df['area_normalized'] = df['area'].str.lower().str.replace('-', '').str.replace(' ', '').str.replace("'", '').fillna('')

    for col in ['area_normalized', 'postcode_district', 'agent_brand', 'postcode_area']:
        means = np.log1p(df.groupby(col)['price_pcm'].transform('mean'))
        counts = df.groupby(col)['price_pcm'].transform('count')
        df[f'{col}_encoded'] = (means * counts + global_mean * smoothing) / (counts + smoothing)

    df['ppsf'] = df['price_pcm'] / df['size_sqft']
    for col in ['area_normalized', 'postcode_district']:
        mean_ppsf = df.groupby(col)['ppsf'].transform('mean')
        df[f'{col}_ppsf_encoded'] = mean_ppsf.fillna(df['ppsf'].mean())

    df['size_postcode_interaction'] = df['size_sqft'] * df['postcode_district_encoded'] / 1000
    df['size_area_interaction'] = df['size_sqft'] * df['area_normalized_encoded'] / 1000
    df['outdoor_location_premium'] = df['has_outdoor_space'] * df['postcode_district_ppsf_encoded']

    # V10: Floor-location interactions
    df['floor_location_premium'] = df['floor_count'] * df['postcode_district_ppsf_encoded'] / 10
    df['multi_floor_central'] = df['is_multi_floor'] * df['center_distance_inverse']
    df['basement_location'] = df['has_basement'] * df['postcode_district_ppsf_encoded']

    # One-hot encode property type
    df = pd.get_dummies(df, columns=['property_type_std'], prefix='type')

    total_features = len([c for c in df.columns if c not in ['price_pcm', 'features', 'description', 'summary', 'room_details']])
    print(f"\n  Total features engineered: {total_features}")

    return df


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


def get_feature_columns(df):
    """Get V10 feature columns."""
    base_features = [
        'bedrooms', 'bathrooms', 'size_sqft', 'log_sqft',
        'sqft_per_bed', 'bath_ratio', 'beds_squared',
        'center_distance_km', 'center_distance_inverse', 'center_distance_squared',
        'center_distance_log', 'tube_distance_km',
        'amenity_score', 'luxury_amenity_score', 'has_outdoor_space',
        'amenity_density', 'weighted_luxury_score',
        'room_diversity', 'non_bedroom_count', 'total_rooms',
        'property_type_encoded', 'prop_type_size',
        'is_short_let', 'is_long_let',
        'postcode_was_inferred', 'is_premium_agent', 'agent_brand_encoded',
    ]

    # V10: Enhanced floor features
    floor_features = [
        'floor_count', 'has_roof_terrace',
        'property_levels_encoded', 'is_single_floor', 'is_duplex', 'is_triplex', 'is_multi_floor',
        'floor_diversity', 'vertical_luxury_score',
        'floor_span_size', 'floor_diversity_size', 'levels_size_interaction',
        'basement_premium', 'basement_beds',
        'multi_floor_beds', 'duplex_size', 'triplex_size',
        'roof_terrace_size',
        'floor_location_premium', 'multi_floor_central', 'basement_location',
        'luxury_floor_count', 'luxury_vertical',
        'has_basement', 'has_lower_ground', 'has_ground', 'has_mezzanine',
        'has_first_floor', 'has_second_floor', 'has_third_floor', 'has_fourth_plus',
    ]

    interaction_features = [
        'short_let_size', 'long_let_size', 'short_let_centrality',
        'center_size_interaction', 'tube_size_interaction',
        'amenity_size_interaction', 'outdoor_size_interaction',
        'luxury_center_interaction', 'premium_agent_size',
        'beds_center_interaction', 'premium_center_interaction',
        'short_let_premium', 'amenity_beds_interaction',
        'central_size_premium',
    ]

    luxury_features = [
        'luxury_size_interaction', 'luxury_bed_interaction', 'luxury_bath_interaction',
        'premium_agent_luxury', 'bath_bed_ratio_luxury', 'bath_bed_ratio_standard',
    ]

    encoding_features = [
        'area_normalized_encoded', 'postcode_district_encoded',
        'postcode_area_encoded',  # agent_brand_encoded already in base
        'area_normalized_ppsf_encoded', 'postcode_district_ppsf_encoded',
        'size_postcode_interaction', 'size_area_interaction',
        'outdoor_location_premium',
    ]

    amenity_cols = [col for col in AMENITY_FEATURES if col in df.columns]
    room_cols = ['has_reception', 'has_kitchen', 'has_dining', 'has_study', 'has_utility']
    type_cols = [col for col in df.columns if col.startswith('type_')]

    all_features = (base_features + floor_features + interaction_features +
                    luxury_features + encoding_features + amenity_cols + room_cols + type_cols)

    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in all_features:
        if f not in seen and f in df.columns:
            seen.add(f)
            unique_features.append(f)

    return unique_features


def train_xgboost_v10(df, quick=False):
    """Train XGBoost with V10 features."""
    print("\n" + "="*70)
    print("TRAINING XGBOOST V10 - ENHANCED FLOOR FEATURES")
    print("="*70)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df['price_pcm'].copy()

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")

    # Default XGBoost params
    default_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }

    if not quick and HAS_OPTUNA:
        print("\n[OPTUNA TUNING]")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 800),
                'max_depth': trial.suggest_int('max_depth', 5, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
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

    print(f"\n[V10 RESULTS]")
    print(f"  R²:         {metrics['r2']:.4f}")
    print(f"  MAE:        £{metrics['mae']:,.0f}")
    print(f"  MAPE:       {metrics['mape']:.1f}%")
    print(f"  Median APE: {metrics['median_ape']:.1f}%")

    # Train on full data for feature importance
    model.fit(X, np.log1p(y))

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n[TOP 20 FEATURES]")
    for _, row in importance.head(20).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:<35} {row['importance']:.4f} {bar}")

    # V10 floor feature importance
    floor_features = [f for f in feature_cols if any(x in f for x in
        ['floor', 'basement', 'duplex', 'triplex', 'multi_floor', 'vertical', 'levels', 'roof_terrace',
         'has_ground', 'has_first', 'has_second', 'has_third', 'has_fourth', 'has_mezzanine', 'has_lower'])]

    floor_importance = importance[importance['feature'].isin(floor_features)]
    print(f"\n[V10 FLOOR FEATURE IMPORTANCE]")
    for _, row in floor_importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:<35} {row['importance']:.4f} {bar}")

    total_floor_importance = floor_importance['importance'].sum()
    print(f"\n  Total floor feature importance: {total_floor_importance:.3f} ({total_floor_importance*100:.1f}%)")

    return model, metrics, importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Skip Optuna tuning')
    args = parser.parse_args()

    # Load and engineer
    df = load_and_clean_data()
    df = engineer_features_v10(df)

    # Train
    model, metrics, importance = train_xgboost_v10(df, quick=args.quick)

    # Version history
    print("\n" + "="*70)
    print("VERSION HISTORY")
    print("="*70)
    print("V1 (baseline):    R²=0.658, MAE=£2,589, MAPE=28.2%")
    print("V2 (Grok/Gemini): R²=0.701, MAE=£1,937, MAPE=23.1%")
    print("V3 (augmented):   R²=0.777, MAE=£1,624, MAPE=19.8%")
    print("V4 (sample wt):   R²=0.780, MAE=£1,598, MAPE=19.3%")
    print("V5 (amenities):   R²=0.773, MAE=£1,650, MAPE=20.1%")
    print("V6 (standardized):R²=0.826, MAE=£1,628, MAPE=20.6%")
    print("V7 (Optuna):      R²=0.834, MAE=£1,584, MAPE=20.3%")
    print("V9 (Grok):        R²=0.839, MAE=£1,560, MAPE=19.6%")
    print(f"V10 (Floor):      R²={metrics['r2']:.3f}, MAE=£{metrics['mae']:,.0f}, MAPE={metrics['mape']:.1f}%")

    v9_r2 = 0.839
    change = (metrics['r2'] - v9_r2) / v9_r2 * 100
    print(f"\nV10 vs V9 R² change: {change:+.1f}%")


if __name__ == '__main__':
    main()
