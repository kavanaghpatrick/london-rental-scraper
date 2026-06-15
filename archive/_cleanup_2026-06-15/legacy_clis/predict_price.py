"""
London Rental Price Predictor
Uses V7 XGBoost model to predict rental prices for individual properties.

Usage:
    # Command line
    python predict_price.py --sqft 850 --beds 2 --postcode SW3 --area Chelsea
    python predict_price.py --sqft 1200 --beds 3 --baths 2 --postcode W8 --area Kensington --furnished
    python predict_price.py --sqft 2000 --beds 3 --postcode SW1 --amenities balcony,porter,gym

    # Interactive mode
    python predict_price.py --interactive

    # From Python
    from predict_price import predict_rent
    price = predict_rent(size_sqft=850, bedrooms=2, postcode='SW3', area='Chelsea')
    price = predict_rent(size_sqft=2000, bedrooms=3, postcode='SW1', amenities=['balcony', 'porter'])
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import argparse
import os
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Get absolute path to database
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'output', 'rentals.db')

# ============ Location Data (from V7 model) ============
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


# ============ Helper Functions ============
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


def get_coords_from_postcode(postcode):
    """Get lat/lon from postcode."""
    pc = postcode.upper().replace(' ', '')
    if pc in POSTCODE_CENTROIDS:
        return POSTCODE_CENTROIDS[pc]
    # Try shorter versions
    for length in [4, 3, 2]:
        if len(pc) >= length:
            pc_short = pc[:length]
            if pc_short in POSTCODE_CENTROIDS:
                return POSTCODE_CENTROIDS[pc_short]
    return (51.5074, -0.1278)  # Default to city center


# ============ Model Training ============
_cached_model = None
_cached_encodings = None


def train_model():
    """Train the V7 model and return it with encoding statistics."""
    global _cached_model, _cached_encodings

    if _cached_model is not None:
        return _cached_model, _cached_encodings

    # Validate database exists
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run scrapers first.")

    print("Training model on clean data (excluding Rightmove)...")

    conn = sqlite3.connect(DB_PATH)
    # V8: Exclude Rightmove (22% outlier rate) and John D Wood (rural properties)
    df = pd.read_sql("""
        SELECT
            bedrooms, bathrooms, size_sqft,
            postcode, area, property_type,
            price_pcm, latitude, longitude, features, source,
            property_type_std, let_type, postcode_normalized,
            postcode_inferred, agent_brand
        FROM listings
        WHERE size_sqft > 0
          AND bedrooms IS NOT NULL
          AND price_pcm > 0
          AND source NOT IN ('rightmove', 'johndwood')
    """, conn)
    conn.close()

    # V8 Quality filters (stricter, based on data audit)
    df['ppsf'] = df['price_pcm'] / df['size_sqft']
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)

    mask1 = (df['ppsf'] >= 2) & (df['ppsf'] <= 30)  # Reasonable £/sqft
    mask2 = sqft_per_bed >= 100  # At least 100 sqft per bedroom
    mask3 = df['size_sqft'] >= 200  # Minimum viable size
    mask4 = df['price_pcm'] <= 50000  # Cap extreme prices

    df = df[mask1 & mask2 & mask3 & mask4].copy()

    # Engineer features (same as V7)
    df['property_type_std'] = df['property_type_std'].fillna('flat')
    df['let_type'] = df['let_type'].fillna('unknown')
    df['is_short_let'] = (df['let_type'] == 'short').astype(int)
    df['is_long_let'] = (df['let_type'] == 'long').astype(int)
    df['postcode_district'] = df['postcode_normalized'].fillna('SW3')
    df['postcode_area'] = df['postcode_district'].str.extract(r'^([A-Z]+)', expand=False).fillna('SW')
    df['postcode_was_inferred'] = df['postcode_inferred'].fillna(0).astype(int)
    df['agent_brand'] = df['agent_brand'].fillna('unknown')
    df['is_premium_agent'] = df['agent_brand'].isin(PREMIUM_AGENTS).astype(int)
    df['area_normalized'] = df['area'].str.lower().str.replace('-', '').str.replace(' ', '').str.replace("'", '').fillna('')

    # Parse amenities from features JSON
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

    amenity_dicts = df['features'].apply(parse_amenities)
    amenity_df = pd.DataFrame(amenity_dicts.tolist())
    for col in AMENITY_FEATURES:
        df[col] = amenity_df[col].values

    df['amenity_score'] = df[AMENITY_FEATURES].sum(axis=1)
    luxury_amenities = ['has_pool', 'has_porter', 'has_gym', 'has_ac']
    df['luxury_amenity_score'] = df[luxury_amenities].sum(axis=1)
    df['has_outdoor_space'] = ((df['has_balcony'] == 1) | (df['has_terrace'] == 1) | (df['has_garden'] == 1)).astype(int)

    # Coordinates
    def get_coords(row):
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or lat == 0:
            coords = get_coords_from_postcode(row['postcode_district'])
            return pd.Series(coords)
        return pd.Series([lat, lon])

    df[['lat_filled', 'lon_filled']] = df.apply(get_coords, axis=1)
    df['tube_distance_km'] = df.apply(lambda r: get_nearest_tube_distance(r['lat_filled'], r['lon_filled']), axis=1)
    df['center_distance_km'] = df.apply(lambda r: get_distance_to_center(r['lat_filled'], r['lon_filled']), axis=1)

    median_tube = df['tube_distance_km'].median()
    median_center = df['center_distance_km'].median()
    df['tube_distance_km'] = df['tube_distance_km'].fillna(median_tube)
    df['center_distance_km'] = df['center_distance_km'].fillna(median_center)

    # Bathrooms
    df['bathrooms'] = df['bathrooms'].fillna(1)

    # Target encodings - store for predictions
    global_mean = np.log1p(df['price_pcm']).mean()
    smoothing = 15

    encodings = {
        'global_mean': global_mean,
        'median_tube': median_tube,
        'median_center': median_center,
    }

    for col in ['area_normalized', 'postcode_district', 'agent_brand', 'postcode_area']:
        group_stats = df.groupby(col)['price_pcm'].agg(['mean', 'count']).reset_index()
        group_stats['log_mean'] = np.log1p(group_stats['mean'])
        group_stats['encoded'] = (group_stats['log_mean'] * group_stats['count'] + global_mean * smoothing) / (group_stats['count'] + smoothing)
        encodings[f'{col}_map'] = dict(zip(group_stats[col], group_stats['encoded']))

    # Price per sqft encodings
    df['ppsf'] = df['price_pcm'] / df['size_sqft']
    mean_ppsf = df['ppsf'].mean()
    encodings['mean_ppsf'] = mean_ppsf

    for col in ['area_normalized', 'postcode_district']:
        ppsf_map = df.groupby(col)['ppsf'].mean().to_dict()
        encodings[f'{col}_ppsf_map'] = ppsf_map

    # Size bins
    size_bins = pd.qcut(df['size_sqft'], q=5, retbins=True)[1]
    encodings['size_bins'] = size_bins

    # Apply encodings to training data
    for col in ['area_normalized', 'postcode_district', 'agent_brand', 'postcode_area']:
        df[f'{col}_encoded'] = df[col].map(encodings[f'{col}_map']).fillna(global_mean)

    for col in ['area_normalized', 'postcode_district']:
        df[f'{col}_ppsf_encoded'] = df[col].map(encodings[f'{col}_ppsf_map']).fillna(mean_ppsf)

    # Feature engineering
    beds_adj = df['bedrooms'].replace(0, 0.5)
    df['size_per_bed'] = df['size_sqft'] / beds_adj
    df['bath_ratio'] = df['bathrooms'] / beds_adj
    df['bed_bath_interaction'] = df['bedrooms'] * df['bathrooms']
    df['size_poly'] = df['size_sqft'] ** 1.5 / 1000
    df['sqft_per_bath'] = df['size_sqft'] / (df['bathrooms'] + 1)
    df['log_sqft'] = np.log1p(df['size_sqft'])
    df['size_squared'] = df['size_sqft'] ** 2 / 100000
    df['sqrt_sqft'] = np.sqrt(df['size_sqft'])
    df['beds_squared'] = df['bedrooms'] ** 2
    df['size_bin'] = pd.cut(df['size_sqft'], bins=size_bins, labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)

    df['short_let_size'] = df['is_short_let'] * df['log_sqft']
    df['long_let_size'] = df['is_long_let'] * df['log_sqft']
    df['center_size_interaction'] = df['center_distance_km'] * df['size_sqft'] / 1000
    df['log_center_distance'] = np.log1p(df['center_distance_km'])
    df['tube_size_interaction'] = df['tube_distance_km'] * df['size_sqft'] / 1000
    df['log_tube_distance'] = np.log1p(df['tube_distance_km'])

    df['size_postcode_interaction'] = df['size_sqft'] * df['postcode_district_encoded'] / 1000
    df['size_area_interaction'] = df['size_sqft'] * df['area_normalized_encoded'] / 1000

    df['beds_center_interaction'] = df['bedrooms'] * (5 - df['center_distance_km']).clip(lower=0)
    df['premium_center_interaction'] = df['is_premium_agent'] * (5 - df['center_distance_km']).clip(lower=0)
    df['short_let_premium'] = df['is_short_let'] * df['is_premium_agent']
    df['premium_agent_size'] = df['is_premium_agent'] * df['log_sqft']

    # Amenity interactions
    df['amenity_size_interaction'] = df['amenity_score'] * df['log_sqft']
    df['outdoor_size_interaction'] = df['has_outdoor_space'] * df['size_sqft'] / 1000
    df['luxury_center_interaction'] = df['luxury_amenity_score'] * (5 - df['center_distance_km']).clip(lower=0)
    df['amenity_beds_interaction'] = df['amenity_score'] * df['bedrooms']

    # Luxury indicators
    df['is_luxury'] = (df['price_pcm'] > 10000).astype(int)
    df['luxury_size_interaction'] = df['is_luxury'] * df['size_sqft'] / 1000

    # One-hot property type
    df = pd.get_dummies(df, columns=['property_type_std'], prefix='type')

    # Feature columns (must match V7 exactly)
    feature_cols = [
        # Core features
        'bedrooms', 'bathrooms', 'size_sqft',
        'size_per_bed', 'bath_ratio', 'bed_bath_interaction',
        'size_poly', 'sqft_per_bath', 'log_sqft',
        'size_squared', 'sqrt_sqft', 'beds_squared',

        # Location encodings
        'area_normalized_encoded', 'postcode_district_encoded', 'postcode_area_encoded',
        'size_postcode_interaction', 'size_area_interaction',
        'area_normalized_ppsf_encoded', 'postcode_district_ppsf_encoded',
        'postcode_was_inferred',

        # Distance features
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

    ] + AMENITY_FEATURES

    # Add property type dummies that exist
    type_cols = [c for c in df.columns if c.startswith('type_')]
    feature_cols.extend(type_cols)
    encodings['type_cols'] = type_cols
    encodings['feature_cols'] = feature_cols

    X = df[feature_cols].fillna(0)
    y = np.log1p(df['price_pcm'])

    # Train XGBoost with V7 Optuna-tuned params
    model = XGBRegressor(
        n_estimators=1573,
        learning_rate=0.0129,
        max_depth=12,
        subsample=0.73,
        colsample_bytree=0.62,
        min_child_weight=2,
        reg_alpha=0.11,
        reg_lambda=2.67,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X, y)

    _cached_model = model
    _cached_encodings = encodings

    print(f"Model trained on {len(df)} properties")
    return model, encodings


def predict_rent(
    size_sqft: int,
    bedrooms: int,
    postcode: str,
    area: str = None,
    bathrooms: int = None,
    property_type: str = 'flat',
    let_type: str = 'long',
    agent: str = None,
    latitude: float = None,
    longitude: float = None,
    amenities: list = None,
    verbose: bool = True
) -> dict:
    """
    Predict monthly rent for a property.

    Args:
        size_sqft: Property size in square feet
        bedrooms: Number of bedrooms (0 for studio)
        postcode: Postcode district (e.g., 'SW3', 'W8', 'NW1')
        area: Area name (e.g., 'Chelsea', 'Kensington') - optional
        bathrooms: Number of bathrooms (defaults to bedrooms if not provided)
        property_type: 'flat', 'house', 'maisonette', 'studio'
        let_type: 'long', 'short', 'unknown'
        agent: Agent name (optional)
        latitude: Latitude coordinate (optional)
        longitude: Longitude coordinate (optional)
        amenities: List of amenities (optional)
        verbose: Print details

    Returns:
        dict with predicted_price, confidence_range, and details
    """
    model, encodings = train_model()

    # Defaults
    if bathrooms is None:
        bathrooms = max(1, bedrooms)

    if area is None:
        area = postcode  # Use postcode as area if not provided

    # Normalize inputs
    postcode_district = postcode.upper().replace(' ', '')
    postcode_area = ''.join([c for c in postcode_district if c.isalpha()])
    area_normalized = area.lower().replace('-', '').replace(' ', '').replace("'", '')

    # Get coordinates
    if latitude and longitude:
        lat, lon = latitude, longitude
    else:
        lat, lon = get_coords_from_postcode(postcode_district)

    tube_distance = get_nearest_tube_distance(lat, lon) or encodings['median_tube']
    center_distance = get_distance_to_center(lat, lon) or encodings['median_center']

    # Determine encodings
    global_mean = encodings['global_mean']

    area_encoded = encodings['area_normalized_map'].get(area_normalized, global_mean)
    postcode_encoded = encodings['postcode_district_map'].get(postcode_district, global_mean)
    postcode_area_encoded = encodings['postcode_area_map'].get(postcode_area, global_mean)
    agent_encoded = encodings['agent_brand_map'].get(agent or 'unknown', global_mean)

    area_ppsf = encodings['area_normalized_ppsf_map'].get(area_normalized, encodings['mean_ppsf'])
    postcode_ppsf = encodings['postcode_district_ppsf_map'].get(postcode_district, encodings['mean_ppsf'])

    is_premium = 1 if agent in PREMIUM_AGENTS else 0
    is_short = 1 if let_type == 'short' else 0
    is_long = 1 if let_type == 'long' else 0

    # Size bin with clipping for out-of-range values
    size_bins = encodings['size_bins']
    # Clip size to training data range to avoid NaN bins
    clipped_sqft = max(size_bins[0], min(size_bins[-1] - 1, size_sqft))
    size_bin = pd.cut([clipped_sqft], bins=size_bins, labels=[0, 1, 2, 3, 4])[0]
    if pd.isna(size_bin):
        size_bin = 2  # Middle bin as fallback

    # Parse amenities input
    amenity_dict = {f: 0 for f in AMENITY_FEATURES}
    if amenities:
        # Map common amenity names to feature names
        amenity_mapping = {
            'balcony': 'has_balcony', 'terrace': 'has_terrace', 'garden': 'has_garden',
            'porter': 'has_porter', 'concierge': 'has_porter',
            'gym': 'has_gym', 'pool': 'has_pool', 'parking': 'has_parking',
            'lift': 'has_lift', 'elevator': 'has_lift',
            'ac': 'has_ac', 'air conditioning': 'has_ac', 'aircon': 'has_ac',
            'furnished': 'has_furnished',
            'high ceilings': 'has_high_ceilings', 'high_ceilings': 'has_high_ceilings',
            'view': 'has_view', 'views': 'has_view',
            'modern': 'has_modern', 'period': 'has_period',
        }
        for amenity in amenities:
            amenity_lower = amenity.lower().strip()
            if amenity_lower in amenity_mapping:
                amenity_dict[amenity_mapping[amenity_lower]] = 1
            elif f'has_{amenity_lower}' in AMENITY_FEATURES:
                amenity_dict[f'has_{amenity_lower}'] = 1

    amenity_score = sum(amenity_dict.values())
    luxury_amenities = ['has_pool', 'has_porter', 'has_gym', 'has_ac']
    luxury_amenity_score = sum(amenity_dict[a] for a in luxury_amenities)
    has_outdoor_space = 1 if (amenity_dict['has_balcony'] or amenity_dict['has_terrace'] or amenity_dict['has_garden']) else 0

    # Feature engineering
    beds_adj = max(0.5, bedrooms)
    log_sqft = np.log1p(size_sqft)
    center_proximity = max(0, 5 - center_distance)

    # Estimate if property is luxury (based on area encoding which correlates with price)
    # Use area encoding as proxy - high encoding means typically expensive area
    is_luxury_estimate = 1 if area_encoded > encodings['global_mean'] * 1.1 else 0

    features = {
        # Core features
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'size_sqft': size_sqft,
        'size_per_bed': size_sqft / beds_adj,
        'bath_ratio': bathrooms / beds_adj,
        'bed_bath_interaction': bedrooms * bathrooms,
        'size_poly': size_sqft ** 1.5 / 1000,
        'sqft_per_bath': size_sqft / (bathrooms + 1),
        'log_sqft': log_sqft,
        'size_squared': size_sqft ** 2 / 100000,
        'sqrt_sqft': np.sqrt(size_sqft),
        'beds_squared': bedrooms ** 2,

        # Location encodings
        'area_normalized_encoded': area_encoded,
        'postcode_district_encoded': postcode_encoded,
        'postcode_area_encoded': postcode_area_encoded,
        'size_postcode_interaction': size_sqft * postcode_encoded / 1000,
        'size_area_interaction': size_sqft * area_encoded / 1000,
        'area_normalized_ppsf_encoded': area_ppsf,
        'postcode_district_ppsf_encoded': postcode_ppsf,
        'postcode_was_inferred': 0,  # User-provided postcodes are not inferred

        # Distance features
        'tube_distance_km': tube_distance,
        'log_tube_distance': np.log1p(tube_distance),
        'tube_size_interaction': tube_distance * size_sqft / 1000,
        'center_distance_km': center_distance,
        'log_center_distance': np.log1p(center_distance),
        'center_size_interaction': center_distance * size_sqft / 1000,

        # Let type
        'is_short_let': is_short,
        'is_long_let': is_long,
        'short_let_size': is_short * log_sqft,
        'long_let_size': is_long * log_sqft,
        'short_let_premium': is_short * is_premium,

        # Size category
        'size_bin': float(size_bin),
        'luxury_size_interaction': is_luxury_estimate * size_sqft / 1000,

        # Agent brand
        'agent_brand_encoded': agent_encoded,
        'is_premium_agent': is_premium,
        'premium_agent_size': is_premium * log_sqft,
        'premium_center_interaction': is_premium * center_proximity,

        # Amenity features
        'amenity_score': amenity_score,
        'luxury_amenity_score': luxury_amenity_score,
        'has_outdoor_space': has_outdoor_space,
        'amenity_size_interaction': amenity_score * log_sqft,
        'outdoor_size_interaction': has_outdoor_space * size_sqft / 1000,
        'luxury_center_interaction': luxury_amenity_score * center_proximity,
        'amenity_beds_interaction': amenity_score * bedrooms,

        # Location interactions
        'beds_center_interaction': bedrooms * center_proximity,
    }

    # Add individual amenity features
    for amenity_feat in AMENITY_FEATURES:
        features[amenity_feat] = amenity_dict[amenity_feat]

    # Add property type dummies
    for col in encodings['type_cols']:
        prop_type = col.replace('type_', '')
        features[col] = 1 if property_type.lower() == prop_type else 0

    # Create DataFrame with correct column order
    X_pred = pd.DataFrame([features])[encodings['feature_cols']]
    X_pred = X_pred.fillna(0)

    # Predict
    log_price = model.predict(X_pred)[0]
    predicted_price = np.expm1(log_price)

    # Confidence range (based on model's typical error: ~10% MAPE)
    low_estimate = predicted_price * 0.90
    high_estimate = predicted_price * 1.10

    # Price per sqft
    price_per_sqft = predicted_price / size_sqft

    result = {
        'predicted_price': round(predicted_price),
        'low_estimate': round(low_estimate),
        'high_estimate': round(high_estimate),
        'price_per_sqft': round(price_per_sqft, 2),
        'inputs': {
            'size_sqft': size_sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'postcode': postcode_district,
            'area': area,
            'property_type': property_type,
            'let_type': let_type,
        },
        'location': {
            'distance_to_center_km': round(center_distance, 2),
            'distance_to_nearest_tube_km': round(tube_distance, 2),
        }
    }

    if verbose:
        print("\n" + "=" * 50)
        print("RENTAL PRICE PREDICTION")
        print("=" * 50)
        print(f"\nProperty Details:")
        print(f"  Size:     {size_sqft:,} sq ft")
        print(f"  Beds:     {bedrooms}")
        print(f"  Baths:    {bathrooms}")
        print(f"  Type:     {property_type}")
        print(f"  Postcode: {postcode_district}")
        print(f"  Area:     {area}")
        print(f"  Let type: {let_type}")
        if agent:
            print(f"  Agent:    {agent} {'(Premium)' if is_premium else ''}")
        if amenity_score > 0:
            active_amenities = [k.replace('has_', '') for k, v in amenity_dict.items() if v == 1]
            print(f"  Amenities: {', '.join(active_amenities)}")
        print(f"\nLocation:")
        print(f"  Distance to center:       {center_distance:.1f} km")
        print(f"  Distance to nearest tube: {tube_distance:.1f} km")
        print(f"\n{'=' * 50}")
        print(f"PREDICTED RENT:  £{predicted_price:,.0f} pcm")
        print(f"{'=' * 50}")
        print(f"  Range: £{low_estimate:,.0f} - £{high_estimate:,.0f} pcm")
        print(f"  Price/sqft: £{price_per_sqft:.2f}")
        print()

    return result


def interactive_mode():
    """Run interactive prediction mode."""
    print("\n" + "=" * 50)
    print("LONDON RENTAL PRICE PREDICTOR")
    print("=" * 50)
    print("\nEnter property details (or 'q' to quit)\n")

    while True:
        try:
            sqft = input("Size (sq ft): ").strip()
            if sqft.lower() == 'q':
                break
            sqft = int(sqft)

            beds = input("Bedrooms (0 for studio): ").strip()
            if beds.lower() == 'q':
                break
            beds = int(beds)

            postcode = input("Postcode district (e.g., SW3, W8): ").strip()
            if postcode.lower() == 'q':
                break

            area = input("Area name (optional, press Enter to skip): ").strip()
            if area.lower() == 'q':
                break
            area = area if area else None

            baths = input("Bathrooms (optional, press Enter for default): ").strip()
            baths = int(baths) if baths and baths.lower() != 'q' else None

            prop_type = input("Property type [flat/house/maisonette] (default: flat): ").strip()
            prop_type = prop_type if prop_type else 'flat'

            amenities_input = input("Amenities (comma-separated, e.g., balcony,porter,gym - or press Enter to skip): ").strip()
            amenities_list = [a.strip() for a in amenities_input.split(',')] if amenities_input else None

            predict_rent(
                size_sqft=sqft,
                bedrooms=beds,
                postcode=postcode,
                area=area,
                bathrooms=baths,
                property_type=prop_type,
                amenities=amenities_list
            )

            print("\n" + "-" * 50)
            another = input("\nPredict another property? (y/n): ").strip().lower()
            if another != 'y':
                break

        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Predict London rental prices')
    parser.add_argument('--sqft', type=int, help='Property size in square feet')
    parser.add_argument('--beds', type=int, help='Number of bedrooms (0 for studio)')
    parser.add_argument('--baths', type=int, help='Number of bathrooms')
    parser.add_argument('--postcode', type=str, help='Postcode district (e.g., SW3, W8)')
    parser.add_argument('--area', type=str, help='Area name (e.g., Chelsea, Kensington)')
    parser.add_argument('--type', type=str, default='flat', help='Property type: flat, house, maisonette')
    parser.add_argument('--let', type=str, default='long', help='Let type: long, short')
    parser.add_argument('--agent', type=str, help='Agent name')
    parser.add_argument('--amenities', type=str, help='Comma-separated amenities (e.g., balcony,porter,gym)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.sqft and args.beds is not None and args.postcode:
        amenities_list = args.amenities.split(',') if args.amenities else None
        predict_rent(
            size_sqft=args.sqft,
            bedrooms=args.beds,
            bathrooms=args.baths,
            postcode=args.postcode,
            area=args.area,
            property_type=args.type,
            let_type=args.let,
            agent=args.agent,
            amenities=amenities_list
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python predict_price.py --sqft 850 --beds 2 --postcode SW3")
        print("  python predict_price.py --sqft 1200 --beds 3 --baths 2 --postcode W8 --area Kensington")
        print("  python predict_price.py --sqft 2000 --beds 3 --postcode SW1 --amenities balcony,porter,gym")
        print("  python predict_price.py --sqft 1500 --beds 3 --postcode W8 --agent 'Knight Frank'")
        print("  python predict_price.py --interactive")


if __name__ == '__main__':
    main()
