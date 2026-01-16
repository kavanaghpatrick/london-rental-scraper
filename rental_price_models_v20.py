"""
London Rental Price Prediction Models V20 - MERGED V16 + V19 FEATURES

Combines:
- V16 (investigations): social housing, postcode adjustment, size anomalies, bathroom signals
- V19 (Chrome): garden squares, prime streets, mews PPSF, type target encoding

Usage:
    python rental_price_models_v20.py
    python rental_price_models_v20.py --quick
    python rental_price_models_v20.py --export  # Export to Chrome extension
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import argparse
import pickle
import re
from pathlib import Path
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

# Import shared constants
try:
    from shared_constants import (
        PPSF_MIN, PPSF_MAX, PRICE_MIN_PCM, SQFT_MIN, SQFT_MAX,
        SQFT_PER_BED_MIN, PRIME_POSTCODES, PREMIUM_AGENTS
    )
except ImportError:
    PPSF_MIN, PPSF_MAX = 3, 30
    PRICE_MIN_PCM = 500
    SQFT_MIN, SQFT_MAX = 150, 10000
    SQFT_PER_BED_MIN = 70
    PRIME_POSTCODES = ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW3', 'NW8']
    # Premium agents - partial match, case-insensitive (sync with xgboost.js)
    PREMIUM_AGENTS = ['knight frank', 'savills', 'harrods', 'sotheby',
                      'beauchamp', 'strutt', 'chestertons', 'carter jonas',
                      'hamptons', 'winkworth', 'marsh']

OUTPUT_DIR = Path('output')

# ==================== V16 DETECTION PATTERNS ====================

# Ultra-prestige streets (from V16 investigation)
ULTRA_PRESTIGE_STREETS = [
    'eaton place', 'eaton square', 'belgrave square', 'chester square',
    'wilton crescent', 'cadogan place', 'cadogan square', 'grosvenor square',
    'upper grosvenor street', 'hyde park gate', 'princes gate', 'hans place',
    'pont street', 'adams row', 'basil street', 'montpelier square',
    'pavilion road', 'sloane street', 'lancelot place', 'trevor square'
]

# Social housing estates (from V16 investigation)
SOCIAL_ESTATE_PATTERNS = [
    r"world'?s?\s*end\s*estate", r"townshend\s*estate", r"hallfield\s*estate",
    r"churchill\s*gardens", r"ebury\s*bridge", r"peabody",
    r"trellick\s*tower", r"lancaster\s*west", r"silchester",
    r"lisson\s*grove", r"penfold\s*place", r"mallory\s*street"
]

# Serviced apartment buildings (from V16)
SERVICED_BUILDINGS = [
    'cheval', 'thorney court', 'the other house',
    'ashburn place', '199 knightsbridge', 'knightsbridge apartments'
]

# Postcode adjustment factors (from V16 investigation)
POSTCODE_ADJUSTMENTS = {
    'W1J': 1.41, 'W1S': 1.47, 'W1K': 1.25, 'W8': 1.20, 'SW1X': 1.13,
    'SW14': 0.83, 'SW17': 0.65, 'SW18': 0.80, 'SW12': 0.64, 'SW13': 0.90
}

# Premium postcodes for social housing detection
PREMIUM_POSTCODES_SOCIAL = ['SW1', 'SW3', 'SW7', 'W1', 'W8', 'NW8']

# ==================== V19 DETECTION PATTERNS ====================

# Garden squares (from V19 Chrome)
GARDEN_SQUARES = [
    'cadogan square', 'belgrave square', 'chester square', 'eaton square',
    'montpelier square', 'brompton square', 'thurloe square', 'lowndes square',
    'trevor square', 'lennox gardens', 'cadogan gardens', 'sloane square',
    'paultons square', 'chelsea square', 'onslow square', 'pelham crescent',
    'egerton crescent', 'egerton gardens', 'ovington square'
]

# Ultra-prime addresses (from V19)
ULTRA_PRIME_ADDRESSES = [
    'belgrave square', 'chester square', 'eaton square', 'wilton crescent',
    'grosvenor square', 'grosvenor crescent', 'upper grosvenor street',
    'park lane', 'hamilton terrace', 'avenue road', 'bishops avenue'
]

# Prime streets (from V19)
PRIME_STREETS = [
    'cadogan square', 'cadogan place', 'cadogan gardens', 'hans place',
    'lennox gardens', 'pont street', 'sloane street', 'draycott place',
    'draycott avenue', 'eaton place', 'eaton terrace', 'montpelier street',
    'brompton square', 'thurloe square', 'ennismore gardens', 'princes gate',
    'hyde park gate', 'kensington palace gardens', 'palace gardens terrace',
    'campden hill', 'holland park', 'phillimore gardens', 'carlyle square',
    'cheyne walk', 'the boltons', 'tregunter road', 'elm park gardens'
]

# Mews PPSF by postcode (from V19)
PC_MEWS_PPSF = {
    'SW1X': 7.00, 'SW1W': 4.36, 'SW1V': 4.51, 'SW1': 5.50,
    'SW10': 8.28, 'SW3': 5.62, 'SW7': 4.14, 'SW5': 3.67, 'SW11': 3.22,
    'W1H': 8.33, 'W1G': 5.06, 'W8': 4.51, 'W2': 4.40, 'W11': 4.11, 'W12': 3.98,
    'NW3': 4.92, 'default': 5.40
}

# House PPSF by postcode (from V19)
PC_HOUSE_PPSF = {
    'SW3': 5.46, 'SW7': 6.02, 'SW1X': 8.28, 'NW8': 5.76, 'SW10': 6.26,
    'NW3': 4.82, 'SW6': 3.52, 'W8': 5.93, 'W11': 6.32, 'SW1W': 6.96,
    'W6': 6.75, 'W2': 5.17, 'W14': 5.33, 'NW1': 5.76, 'SW18': 3.42,
    'SW5': 5.50, 'SW11': 5.00, 'W10': 5.00, 'default': 5.56
}

# Flat PPSF by postcode (from V19)
PC_FLAT_PPSF = {
    'SW3': 6.50, 'SW7': 6.20, 'SW1X': 7.80, 'NW8': 5.50, 'SW10': 5.80,
    'NW3': 5.20, 'SW6': 4.50, 'W8': 6.80, 'W11': 6.00, 'SW1W': 6.50,
    'W6': 5.50, 'W2': 5.80, 'W14': 5.20, 'NW1': 5.50, 'SW5': 5.00,
    'SW11': 4.80, 'W10': 4.50, 'default': 5.25
}

OVERALL_HOUSE_PPSF = 5.5635
OVERALL_FLAT_PPSF = 5.2459
OVERALL_MEWS_PPSF = 5.40
OVERALL_PPSF = 5.2093

# Property type lists (from V19)
HOUSE_TYPES = ['house', 'terraced', 'detached', 'semi-detached', 'town house', 'cottage', 'end of terrace', 'link detached']
FLAT_TYPES = ['flat', 'apartment', 'studio', 'penthouse', 'maisonette', 'duplex', 'ground flat']

# Size quintile boundaries
SIZE_QUINTILES = [0, 484, 635, 818, 1141, float('inf')]

# ==================== V20+ PRESTIGE LOCATION PPSF ====================
# Location-specific PPSF targets (from data analysis)
# Much more accurate than binary is_garden_square

PRESTIGE_LOCATION_PPSF = {
    'draycott avenue': 18.28,
    'trevor square': 13.24,
    'hyde park gate': 12.88,
    'wilton crescent': 11.23,
    'cadogan square': 10.99,
    'lowndes square': 9.11,
    'thurloe square': 8.14,
    'pont street': 7.92,
    'princes gate': 7.85,
    'ennismore gardens': 7.54,
    'cadogan place': 6.93,
    'montpelier square': 6.91,
    'cadogan gardens': 6.78,
    'lennox gardens': 6.76,
    'eaton square': 6.72,
    'hans place': 6.60,
    'sloane square': 6.26,
    'onslow square': 6.19,
    'palace gardens terrace': 6.11,
    'chester square': 6.08,
    'sloane street': 6.07,
    'cheyne walk': 6.07,
    'egerton gardens': 5.92,
    'the boltons': 5.92,
    'brompton square': 5.47,
    'draycott place': 5.43,
    'ovington square': 4.77,
    'holland park': 4.63,
    'elm park gardens': 4.48,
    'belgrave square': 6.76,  # Only 1 sample but known ultra-prime
    'default': 5.42,
}

# Prestige tier classification
PRESTIGE_LOCATION_TIER = {
    # Ultra tier (>80% premium): 4
    'trevor square': 4, 'cadogan square': 4, 'wilton crescent': 4,
    'hyde park gate': 4, 'draycott avenue': 4,
    # High tier (40-80% premium): 3
    'lowndes square': 3, 'thurloe square': 3, 'pont street': 3,
    'princes gate': 3, 'cadogan place': 3, 'hans place': 3,
    'sloane street': 3, 'the boltons': 3,
    # Premium tier (20-40% premium): 2
    'ennismore gardens': 2, 'montpelier square': 2, 'cadogan gardens': 2,
    'lennox gardens': 2, 'eaton square': 2, 'belgrave square': 2,
    'sloane square': 2, 'palace gardens terrace': 2, 'chester square': 2,
    'cheyne walk': 2, 'draycott place': 2,
    # Standard tier (<20% premium): 1
    'onslow square': 1, 'egerton gardens': 1, 'brompton square': 1,
    'ovington square': 1, 'holland park': 1, 'elm park gardens': 1,
}

# Tier multipliers (for addresses not in specific lookup)
PRESTIGE_TIER_MULTIPLIER = {
    4: 2.03,  # Ultra: +103% premium
    3: 1.33,  # High: +33% premium
    2: 1.13,  # Premium: +13% premium
    1: 1.01,  # Standard: +1% premium
    0: 1.00,  # None
}

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
    'Canary Wharf': (51.5033, -0.0181),
}

POSTCODE_CENTROIDS = {
    'SW1': (51.4970, -0.1400), 'SW3': (51.4900, -0.1680), 'SW5': (51.4920, -0.1940),
    'SW7': (51.4950, -0.1780), 'SW10': (51.4830, -0.1820), 'SW11': (51.4650, -0.1650),
    'W1': (51.5150, -0.1450), 'W2': (51.5150, -0.1780), 'W8': (51.5010, -0.1920),
    'W11': (51.5150, -0.2050), 'W14': (51.4950, -0.2100),
    'NW1': (51.5350, -0.1550), 'NW3': (51.5550, -0.1780), 'NW8': (51.5330, -0.1750),
}

CITY_CENTER = (51.5074, -0.1278)

AMENITY_FEATURES = [
    'has_balcony', 'has_terrace', 'has_garden', 'has_porter',
    'has_gym', 'has_pool', 'has_parking', 'has_lift', 'has_ac',
    'has_furnished', 'has_high_ceilings', 'has_view',
    'has_modern', 'has_period', 'has_roof_terrace'
]


# ==================== HELPER FUNCTIONS ====================

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
    amenities = {f: 0 for f in AMENITY_FEATURES}

    text = ''
    if features_str and not pd.isna(features_str):
        text += str(features_str).lower() + ' '
    if description_str and not pd.isna(description_str):
        text += str(description_str).lower()

    try:
        if features_str and not pd.isna(features_str):
            parsed = json.loads(features_str)
            if isinstance(parsed, dict):
                for key in AMENITY_FEATURES:
                    if parsed.get(key):
                        amenities[key] = 1
    except (json.JSONDecodeError, TypeError):
        pass

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


# ==================== DETECTION FUNCTIONS ====================

def is_short_let_or_serviced(row):
    """Detect short lets and serviced apartments to EXCLUDE from training."""
    property_type = str(row.get('property_type', '')).lower()
    address = str(row.get('address', '')).lower()
    ppsf = row.get('ppsf', 0) or 0

    if 'short let' in property_type or 'serviced' in property_type:
        return True
    if any(kw in address for kw in SERVICED_BUILDINGS):
        return True
    if ppsf > 20:
        return True
    return False


def is_likely_social_housing(row):
    """Detect social housing / affordable housing (V16)."""
    address = str(row.get('address', '')).lower()
    ppsf = row.get('ppsf', 0) or 0
    postcode_district = str(row.get('postcode_district', ''))

    for pattern in SOCIAL_ESTATE_PATTERNS:
        if re.search(pattern, address):
            return 1

    if any(postcode_district.startswith(p) for p in PREMIUM_POSTCODES_SOCIAL):
        if ppsf < 3.5:
            return 1
    return 0


def is_garden_square(address):
    """Detect prestigious garden square (V19)."""
    address_lower = str(address).lower()
    return int(any(sq in address_lower for sq in GARDEN_SQUARES))


def is_ultra_prime_address(address):
    """Detect ultra-prime address (V19)."""
    address_lower = str(address).lower()
    return int(any(addr in address_lower for addr in ULTRA_PRIME_ADDRESSES))


def is_prime_street(address):
    """Detect prime street (V19)."""
    address_lower = str(address).lower()
    return int(any(st in address_lower for st in PRIME_STREETS))


def is_ultra_luxury_address_v16(address):
    """Detect ultra-prestige street addresses (V16)."""
    address_lower = str(address).lower()
    return int(any(street in address_lower for street in ULTRA_PRESTIGE_STREETS))


def get_postcode_adjustment(postcode_district):
    """Get adjustment factor for postcode micro-location bias (V16)."""
    return POSTCODE_ADJUSTMENTS.get(postcode_district, 1.0)


def get_prestige_location(address):
    """Find the prestige location in address (V20+)."""
    address_lower = str(address).lower()
    for location in PRESTIGE_LOCATION_PPSF.keys():
        if location != 'default' and location in address_lower:
            return location
    return None


def get_prestige_location_ppsf(address):
    """Get location-specific PPSF target (V20+)."""
    location = get_prestige_location(address)
    if location:
        return PRESTIGE_LOCATION_PPSF.get(location, PRESTIGE_LOCATION_PPSF['default'])
    return PRESTIGE_LOCATION_PPSF['default']


def get_prestige_tier(address):
    """Get prestige tier (0-4) for address (V20+)."""
    location = get_prestige_location(address)
    if location:
        return PRESTIGE_LOCATION_TIER.get(location, 0)
    return 0


def get_prestige_multiplier(address):
    """Get prestige multiplier based on tier (V20+)."""
    tier = get_prestige_tier(address)
    return PRESTIGE_TIER_MULTIPLIER.get(tier, 1.0)


def is_mews(row):
    """Detect mews property (V19)."""
    pt = str(row.get('property_type_std', '')).lower()
    addr = str(row.get('address', '')).lower()
    return int('mews' in pt or 'mews' in addr)


def is_house_type(property_type):
    """Check if property is a house type (V19)."""
    pt = str(property_type).lower()
    if 'mews' in pt:
        return 0
    return int(any(h in pt for h in HOUSE_TYPES))


def is_flat_type(property_type):
    """Check if property is a flat type (V19)."""
    pt = str(property_type).lower()
    return int(any(f in pt for f in FLAT_TYPES))


def get_type_ppsf_target(is_house, is_flat, is_mews):
    """Get type-based PPSF target (V19)."""
    if is_mews:
        return OVERALL_MEWS_PPSF
    if is_house:
        return OVERALL_HOUSE_PPSF
    if is_flat:
        return OVERALL_FLAT_PPSF
    return OVERALL_PPSF


def get_pc_type_ppsf(postcode_district, is_house, is_flat, is_mews):
    """Get postcode × type PPSF (V19)."""
    if is_mews:
        return PC_MEWS_PPSF.get(postcode_district, PC_MEWS_PPSF['default'])
    if is_house:
        return PC_HOUSE_PPSF.get(postcode_district, PC_HOUSE_PPSF['default'])
    if is_flat:
        return PC_FLAT_PPSF.get(postcode_district, PC_FLAT_PPSF['default'])
    return OVERALL_PPSF


def get_size_bin(sqft):
    """Get size quintile bin (V19)."""
    for i, boundary in enumerate(SIZE_QUINTILES[1:], 0):
        if sqft < boundary:
            return i
    return 4


def is_furnished(description):
    """Detect furnished vs unfurnished from description."""
    if not description or pd.isna(description):
        return None
    desc = str(description).lower()
    if 'unfurnished' in desc or 'un-furnished' in desc:
        return 0
    if 'furnished' in desc:
        return 1
    return None


def has_refurb_keywords(description):
    """Detect recently refurbished properties (V16)."""
    if not description or pd.isna(description):
        return 0
    desc = str(description).lower()
    keywords = ['refurbished', 'newly decorated', 'newly renovated',
                'brand new', 'just completed', 'newly fitted']
    return 1 if any(kw in desc for kw in keywords) else 0


# ==================== DATA LOADING ====================

def load_and_clean_data():
    """Load data and apply filtering."""
    import os

    postgres_url = os.environ.get('POSTGRES_URL')

    if postgres_url:
        import psycopg2
        conn = psycopg2.connect(postgres_url)
        query = """
            SELECT
                bedrooms, bathrooms, size_sqft,
                postcode, area, property_type, address,
                price_pcm, latitude, longitude, features, description, source,
                agent_name, price_period,
                floor_count, has_roof_terrace, has_basement, has_ground,
                has_first_floor, has_second_floor, has_third_floor, has_fourth_plus
            FROM listings
            WHERE size_sqft > 0 AND bedrooms IS NOT NULL AND price_pcm > 0
            AND is_active = 1
            AND (description NOT ILIKE '%short let%' OR description IS NULL)
            AND (price_period = 'pcm' OR price_period IS NULL OR price_period = '')
        """
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Loaded from Postgres")
        df['property_type_std'] = df['property_type'].fillna('flat').str.lower()
        df['let_type'] = 'long'
        df['postcode_normalized'] = df['postcode'].str.extract(r'^([A-Z]+\d+[A-Z]?)', expand=False)
        df['agent_brand'] = df['agent_name'].fillna('').apply(
            lambda x: next((a for a in PREMIUM_AGENTS if a.lower() in x.lower()), 'unknown')
        )
    else:
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
            AND is_active = 1
            AND (is_short_let = 0 OR is_short_let IS NULL)
        """
        conn = sqlite3.connect('output/rentals.db')
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Loaded from SQLite")

    print(f"Raw data: {len(df)} records")

    df['ppsf'] = df['price_pcm'] / df['size_sqft']

    # Quality filters
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    mask1 = sqft_per_bed < SQFT_PER_BED_MIN
    mask2 = df['price_pcm'] > 100000
    mask3 = df['size_sqft'] < SQFT_MIN
    mask4 = df['size_sqft'] > SQFT_MAX
    mask5 = df['ppsf'] < PPSF_MIN
    mask6 = df['ppsf'] > PPSF_MAX
    mask7 = df['price_pcm'] < PRICE_MIN_PCM

    df_clean = df[~(mask1 | mask2 | mask3 | mask4 | mask5 | mask6 | mask7)].copy()
    print(f"After quality filters: {len(df_clean)} records")

    # Exclude short lets
    before_short_let = len(df_clean)
    short_let_mask = df_clean.apply(is_short_let_or_serviced, axis=1)
    df_clean = df_clean[~short_let_mask]
    print(f"After excluding short lets: {len(df_clean)} ({before_short_let - len(df_clean)} removed)")

    # Dedupe cross-source
    df_clean['postcode_district'] = df_clean['postcode'].str.extract(r'^([A-Z]+\d+[A-Z]?)', expand=False)
    source_priority = {'savills': 0, 'knightfrank': 1, 'chestertons': 2, 'foxtons': 3, 'rightmove': 4}
    df_clean['source_rank'] = df_clean['source'].map(source_priority).fillna(5)
    df_clean = df_clean.sort_values('source_rank')
    before_dedupe = len(df_clean)
    df_clean = df_clean.drop_duplicates(
        subset=['price_pcm', 'size_sqft', 'bedrooms', 'postcode_district'],
        keep='first'
    )
    print(f"After dedupe: {len(df_clean)} ({before_dedupe - len(df_clean)} cross-source dupes removed)")

    df_clean = df_clean.drop(columns=['source_rank'])

    return df_clean


# ==================== V20 FEATURE ENGINEERING ====================

def engineer_features_v20(df):
    """V20 feature engineering - merged V16 + V19."""
    print("\n[FEATURE ENGINEERING V20]")

    # ========== CORE FEATURES ==========
    df['bathrooms'] = df['bathrooms'].fillna(1)
    beds_adj = df['bedrooms'].replace(0, 0.5)

    # ========== SIZE FEATURES (V16 + V19) ==========
    df['log_sqft'] = np.log1p(df['size_sqft'])
    df['size_squared'] = df['size_sqft'] ** 2 / 100000
    df['sqrt_sqft'] = np.sqrt(df['size_sqft'])
    df['size_per_bed'] = df['size_sqft'] / beds_adj
    df['beds_squared'] = df['bedrooms'] ** 2
    df['size_bin'] = df['size_sqft'].apply(get_size_bin)

    # V16 size anomaly flags
    df['is_tiny'] = (df['size_sqft'] < 400).astype(int)
    df['is_huge'] = (df['size_sqft'] >= 3000).astype(int)
    print(f"  Tiny properties (<400 sqft): {df['is_tiny'].sum()}")
    print(f"  Huge properties (>3000 sqft): {df['is_huge'].sum()}")

    # ========== BATHROOM FEATURES (V16) ==========
    df['bath_ratio'] = df['bathrooms'] / beds_adj
    df['has_ensuite_each'] = (df['bath_ratio'] >= 1).astype(int)
    df['high_bathroom_count'] = (df['bathrooms'] >= 4).astype(int)
    df['excess_bathrooms'] = (df['bathrooms'] - df['bedrooms']).clip(lower=0)
    df['bed_bath_interaction'] = df['bedrooms'] * df['bathrooms']
    print(f"  High bathroom count (4+): {df['high_bathroom_count'].sum()}")

    # ========== SOCIAL HOUSING (V16) ==========
    df['is_social_housing'] = df.apply(is_likely_social_housing, axis=1)
    print(f"  Likely social housing: {df['is_social_housing'].sum()}")

    # ========== ADDRESS PREMIUM FEATURES (V16 + V19) ==========
    df['address'] = df['address'].fillna('')

    # V16 ultra-luxury
    df['is_ultra_luxury_address'] = df['address'].apply(is_ultra_luxury_address_v16)

    # V19 garden square, ultra-prime, prime street
    df['is_garden_square'] = df['address'].apply(is_garden_square)
    df['is_ultra_prime_address'] = df['address'].apply(is_ultra_prime_address)
    df['is_prime_street'] = df['address'].apply(is_prime_street)

    # Address prestige score (legacy)
    df['address_prestige'] = df['is_ultra_prime_address'] * 3 + df['is_garden_square'] * 2 + df['is_prime_street']

    # V20+ Prestige location features (more accurate than binary is_garden_square)
    df['prestige_location_ppsf'] = df['address'].apply(get_prestige_location_ppsf)
    df['prestige_tier'] = df['address'].apply(get_prestige_tier)
    df['prestige_multiplier'] = df['address'].apply(get_prestige_multiplier)
    df['log_prestige_expected_price'] = np.log1p(df['prestige_location_ppsf'] * df['size_sqft'])
    df['prestige_ppsf_ratio'] = df['prestige_location_ppsf'] / PRESTIGE_LOCATION_PPSF['default']

    # Count prestige locations
    prestige_count = (df['prestige_tier'] > 0).sum()
    ultra_prestige = (df['prestige_tier'] == 4).sum()
    print(f"  Prestige locations detected: {prestige_count} (ultra: {ultra_prestige})")

    print(f"  Garden squares: {df['is_garden_square'].sum()}")
    print(f"  Ultra-prime addresses: {df['is_ultra_prime_address'].sum()}")
    print(f"  Prime streets: {df['is_prime_street'].sum()}")

    # ========== POSTCODE FEATURES (V16 + V19) ==========
    df['postcode_district'] = df['postcode_normalized'].fillna('SW3')
    df['postcode_adjustment'] = df['postcode_district'].apply(get_postcode_adjustment)
    df['postcode_area'] = df['postcode_district'].str.extract(r'^([A-Z]+)', expand=False).fillna('SW')
    df['is_prime_postcode'] = df['postcode_district'].apply(
        lambda x: int(any(x.startswith(p) for p in PRIME_POSTCODES))
    )
    print(f"  Properties with postcode adjustment: {(df['postcode_adjustment'] != 1.0).sum()}")

    # ========== PROPERTY TYPE FEATURES (V16 + V19) ==========
    df['property_type_std'] = df['property_type_std'].fillna('flat')
    pt = df['property_type_std'].str.lower()

    # V19 type classifications
    df['is_mews'] = df.apply(is_mews, axis=1)
    df['is_house'] = pt.apply(is_house_type)
    df['is_flat'] = pt.apply(is_flat_type)
    df['is_large_house'] = ((df['is_house'] == 1) & (df['size_sqft'] > 2000)).astype(int)

    # V16 type refinements
    df['is_terraced'] = pt.isin(['terraced', 'town house']).astype(int)
    df['is_penthouse'] = (pt == 'penthouse').astype(int)
    df['is_studio'] = ((df['bedrooms'] == 1) & (df['size_sqft'] < 400)).astype(int)
    df['is_houseboat'] = (pt == 'house boat').astype(int)
    df['is_duplex_maisonette'] = pt.isin(['duplex', 'maisonette']).astype(int)

    # Penthouse from keywords
    df['is_penthouse'] = df['is_penthouse'] | df['address'].str.contains('penthouse', case=False, na=False)
    df['is_penthouse'] = df['is_penthouse'].astype(int)

    print(f"  Mews: {df['is_mews'].sum()}")
    print(f"  Houses: {df['is_house'].sum()}")
    print(f"  Flats: {df['is_flat'].sum()}")
    print(f"  Terraced: {df['is_terraced'].sum()}")
    print(f"  Penthouses: {df['is_penthouse'].sum()}")

    # ========== TYPE PPSF TARGET ENCODING (V19) ==========
    df['type_ppsf_target'] = df.apply(
        lambda r: get_type_ppsf_target(r['is_house'], r['is_flat'], r['is_mews']), axis=1
    )
    df['log_type_expected_price'] = np.log1p(df['type_ppsf_target'] * df['size_sqft'])

    df['pc_type_ppsf'] = df.apply(
        lambda r: get_pc_type_ppsf(r['postcode_district'], r['is_house'], r['is_flat'], r['is_mews']), axis=1
    )
    df['log_pc_type_expected_price'] = np.log1p(df['pc_type_ppsf'] * df['size_sqft'])

    # ========== FLOOR FEATURES (V16) ==========
    df['floor_count'] = df['floor_count'].fillna(0)
    df['has_basement'] = df['has_basement'].fillna(0)
    df['has_ground'] = df['has_ground'].fillna(0)
    df['has_first_floor'] = df['has_first_floor'].fillna(0)
    df['has_second_floor'] = df['has_second_floor'].fillna(0)
    df['has_third_floor'] = df['has_third_floor'].fillna(0)
    df['has_fourth_plus'] = df['has_fourth_plus'].fillna(0)
    df['is_multi_floor'] = (df['floor_count'] >= 2).astype(int)
    df['floor_size_interaction'] = df['floor_count'] * df['size_sqft'] / 1000

    # Floor type keywords
    df['is_garden_flat'] = df['address'].str.contains('garden flat', case=False, na=False).astype(int)
    df['is_basement_flat'] = df['address'].str.contains('basement flat|basement apartment', case=False, na=False).astype(int)
    df['is_ground_floor'] = (df['has_ground'] == 1).astype(int)
    print(f"  Garden flats: {df['is_garden_flat'].sum()}")
    print(f"  Basement flats: {df['is_basement_flat'].sum()}")

    # ========== CONDITION FEATURES (V16 + V18) ==========
    df['description'] = df['description'].fillna('')

    # V18 furnished detection
    def detect_furnished(desc):
        if not desc or pd.isna(desc):
            return 0, 0, 0
        desc_lower = str(desc).lower()
        if 'unfurnished' in desc_lower:
            return 0, 1, 0
        if 'part furnished' in desc_lower or 'part-furnished' in desc_lower:
            return 0, 0, 1
        if 'furnished' in desc_lower:
            return 1, 0, 0
        return 0, 0, 0

    furnished_data = df['description'].apply(detect_furnished)
    df['is_furnished_explicit'] = furnished_data.apply(lambda x: x[0])
    df['is_unfurnished'] = furnished_data.apply(lambda x: x[1])
    df['is_part_furnished'] = furnished_data.apply(lambda x: x[2])

    # V16 refurbishment
    df['has_refurb_keywords'] = df['description'].apply(has_refurb_keywords)

    # Let type
    df['let_type'] = df['let_type'].fillna('unknown')
    df['is_long_let'] = (df['let_type'] == 'long').astype(int)
    df['is_short_let'] = (df['let_type'] == 'short').astype(int)
    print(f"  Has refurb keywords: {df['has_refurb_keywords'].sum()}")

    # ========== AMENITY FEATURES ==========
    amenity_dicts = df.apply(lambda r: parse_amenities(r['features'], r.get('description', '')), axis=1)
    amenity_df = pd.DataFrame(amenity_dicts.tolist())
    for col in AMENITY_FEATURES:
        df[col] = amenity_df[col].values

    df['amenity_score'] = df[AMENITY_FEATURES].sum(axis=1)
    df['premium_amenity_count'] = df[['has_pool', 'has_porter', 'has_gym', 'has_ac']].sum(axis=1)
    df['has_outdoor_space'] = ((df['has_balcony'] == 1) | (df['has_terrace'] == 1) |
                               (df['has_garden'] == 1) | (df['has_roof_terrace'] == 1)).astype(int)

    # ========== AGENT FEATURES ==========
    df['agent_brand'] = df['agent_brand'].fillna('unknown')
    # Use case-insensitive partial matching (sync with xgboost.js PREMIUM_AGENTS logic)
    df['is_premium_agent'] = df['agent_brand'].apply(
        lambda x: 1 if any(agent in x.lower() for agent in PREMIUM_AGENTS) else 0
    )
    source_map = {'savills': 4, 'knightfrank': 4, 'chestertons': 3, 'foxtons': 2, 'rightmove': 1}
    df['source_quality'] = df['source'].map(source_map).fillna(2)

    # ========== COORDINATE FEATURES ==========
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

    df['log_center_distance'] = np.log1p(df['center_distance_km'])
    df['log_tube_distance'] = np.log1p(df['tube_distance_km'])
    df['center_distance_inv'] = 1 / (1 + df['center_distance_km'])

    # ========== INTERACTION FEATURES (MERGED) ==========
    # V19 interactions
    df['size_x_central'] = df['size_sqft'] * df['center_distance_inv'] / 100
    df['size_x_prime'] = df['size_sqft'] * df['is_prime_postcode'] / 1000
    df['beds_x_central'] = df['bedrooms'] * df['center_distance_inv']
    df['premium_agent_size'] = df['is_premium_agent'] * df['log_sqft']
    df['amenity_x_central'] = df['amenity_score'] * df['center_distance_inv']
    df['outdoor_x_prime'] = df['has_outdoor_space'] * df['is_prime_postcode']

    # V19 type/address interactions
    df['garden_square_size'] = df['is_garden_square'] * df['log_sqft']
    df['ultra_prime_size'] = df['is_ultra_prime_address'] * df['size_sqft'] / 1000
    df['prime_street_size'] = df['is_prime_street'] * df['log_sqft']
    df['prestige_x_size'] = df['address_prestige'] * df['log_sqft']

    # V20+ prestige location interactions (much more accurate)
    df['prestige_tier_x_size'] = df['prestige_tier'] * df['log_sqft']
    df['prestige_ppsf_x_sqft'] = df['prestige_location_ppsf'] * df['size_sqft'] / 1000
    df['prestige_multiplier_x_type'] = df['prestige_multiplier'] * df['type_ppsf_target']

    df['house_size_interaction'] = df['is_house'] * df['log_sqft']
    df['flat_size_interaction'] = df['is_flat'] * df['log_sqft']
    df['large_house_size'] = df['is_large_house'] * df['size_sqft'] / 1000
    df['mews_size_interaction'] = df['is_mews'] * df['log_sqft']
    df['mews_x_prime'] = df['is_mews'] * df['is_prime_postcode']

    # V18 furnished interactions
    df['furnished_x_prime'] = df['is_furnished_explicit'] * df['is_prime_postcode']
    df['furnished_x_central'] = df['is_furnished_explicit'] * df['center_distance_inv']
    df['unfurnished_discount'] = df['is_unfurnished'] * df['log_sqft']

    # V16 luxury interactions
    df['luxury_address_size'] = df['is_ultra_luxury_address'] * df['log_sqft']
    df['luxury_bathroom_size'] = df['high_bathroom_count'] * df['log_sqft']
    df['penthouse_size'] = df['is_penthouse'] * df['log_sqft']
    df['terraced_location'] = df['is_terraced'] * df['is_prime_postcode']

    # Short let interactions
    df['short_let_x_central'] = df['is_short_let'] * df['center_distance_inv']
    df['short_let_size'] = df['is_short_let'] * df['log_sqft']

    # ========== POSTCODE ENCODING ==========
    pc_counts = df['postcode_district'].value_counts()
    df['postcode_freq'] = df['postcode_district'].map(pc_counts) / len(df)

    area_counts = df['postcode_area'].value_counts()
    df['postcode_area_freq'] = df['postcode_area'].map(area_counts) / len(df)

    # Property type numeric
    type_map = {'studio': 0, 'flat': 1, 'apartment': 1, 'maisonette': 2, 'house': 3, 'penthouse': 4, 'town house': 3}
    df['property_type_num'] = df['property_type_std'].map(type_map).fillna(1)

    # One-hot encode top postcodes
    top_postcodes = ['SW3', 'SW7', 'W8', 'W2', 'SW5', 'SW11', 'SW10', 'NW8', 'W11',
                     'SW1X', 'NW3', 'SW1W', 'W14', 'NW1', 'W10']
    for pc in top_postcodes:
        df[f'pc_{pc}'] = (df['postcode_district'] == pc).astype(int)

    # One-hot property types (with dtype=int for XGBoost compatibility)
    df = pd.get_dummies(df, columns=['property_type_std'], prefix='type', drop_first=False, dtype=int)

    feature_count = len([c for c in df.columns if c not in
                        ['price_pcm', 'ppsf', 'features', 'description', 'postcode', 'area',
                         'property_type', 'address', 'postcode_normalized', 'agent_brand',
                         'let_type', 'postcode_district', 'postcode_area', 'source',
                         'latitude', 'longitude', 'lat_filled', 'lon_filled', 'postcode_inferred']])
    print(f"\n  Total features: {feature_count}")

    return df


def get_feature_columns_v20(df):
    """Get V20 feature columns - merged V16 + V19."""
    feature_cols = [
        # Core
        'bedrooms', 'bathrooms', 'size_sqft',

        # Size (merged)
        'log_sqft', 'sqrt_sqft', 'size_squared', 'size_per_bed',
        'beds_squared', 'size_bin',
        'is_tiny', 'is_huge',  # V16

        # Bathroom (V16)
        'bath_ratio', 'has_ensuite_each', 'high_bathroom_count',
        'excess_bathrooms', 'bed_bath_interaction',

        # Location
        'tube_distance_km', 'log_tube_distance',
        'center_distance_km', 'log_center_distance', 'center_distance_inv',
        'is_prime_postcode', 'postcode_freq', 'postcode_area_freq',
        'postcode_adjustment',  # V16

        # V16 + V19 detection features
        'is_social_housing',  # V16
        'is_ultra_luxury_address',  # V16
        'is_garden_square', 'is_ultra_prime_address', 'is_prime_street',  # V19
        'address_prestige',  # V19

        # V20+ Prestige location features (data-driven PPSF targets)
        'prestige_location_ppsf', 'prestige_tier', 'prestige_multiplier',
        'log_prestige_expected_price', 'prestige_ppsf_ratio',

        # Property type (merged)
        'is_mews', 'is_house', 'is_flat', 'is_large_house',  # V19
        'is_terraced', 'is_penthouse', 'is_studio', 'is_houseboat', 'is_duplex_maisonette',  # V16
        'property_type_num',

        # Type target encoding (V19)
        'type_ppsf_target', 'log_type_expected_price',
        'pc_type_ppsf', 'log_pc_type_expected_price',

        # Floor (V16)
        'floor_count', 'is_multi_floor', 'floor_size_interaction',
        'has_basement', 'has_ground', 'has_first_floor',
        'has_second_floor', 'has_third_floor', 'has_fourth_plus',
        'is_garden_flat', 'is_basement_flat', 'is_ground_floor',

        # Condition (V16 + V18)
        'is_furnished_explicit', 'is_unfurnished', 'is_part_furnished',  # V18
        'has_refurb_keywords', 'is_long_let',  # V16
        'is_short_let',

        # Agent
        'is_premium_agent', 'premium_agent_size', 'source_quality',

        # Amenities
        'amenity_score', 'premium_amenity_count', 'has_outdoor_space',
        'amenity_x_central', 'outdoor_x_prime',

        # Interactions (merged)
        'size_x_central', 'size_x_prime', 'beds_x_central',
        'garden_square_size', 'ultra_prime_size', 'prime_street_size',  # V19
        'prestige_x_size',  # V19
        'prestige_tier_x_size', 'prestige_ppsf_x_sqft', 'prestige_multiplier_x_type',  # V20+
        'house_size_interaction', 'flat_size_interaction', 'large_house_size',  # V19
        'mews_size_interaction', 'mews_x_prime',  # V19
        'furnished_x_prime', 'furnished_x_central', 'unfurnished_discount',  # V18
        'luxury_address_size', 'luxury_bathroom_size',  # V16
        'penthouse_size', 'terraced_location',  # V16
        'short_let_x_central', 'short_let_size',
    ]

    # Add amenity columns
    feature_cols.extend(AMENITY_FEATURES)

    # Add one-hot encoded columns
    feature_cols.extend([c for c in df.columns if c.startswith('type_')])
    feature_cols.extend([c for c in df.columns if c.startswith('pc_')])

    # Filter to existing columns and remove duplicates while preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if c in df.columns and not (c in seen or seen.add(c))]

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


def analyze_v20_segments(df, preds, actual):
    """Analyze error by V20 segments."""
    print("\n  V20 Segment Analysis:")

    segments = [
        ('Social housing', df['is_social_housing'] == 1),
        ('Ultra-luxury address (V16)', df['is_ultra_luxury_address'] == 1),
        ('Garden square (V19)', df['is_garden_square'] == 1),
        ('Ultra-prime (V19)', df['is_ultra_prime_address'] == 1),
        ('Prime street (V19)', df['is_prime_street'] == 1),
        ('Mews (V19)', df['is_mews'] == 1),
        ('Tiny (<400 sqft)', df['is_tiny'] == 1),
        ('Huge (>3000 sqft)', df['is_huge'] == 1),
        ('Penthouse', df['is_penthouse'] == 1),
        ('Terraced', df['is_terraced'] == 1),
        ('High bathroom (4+)', df['high_bathroom_count'] == 1),
    ]

    for name, mask in segments:
        if mask.sum() > 0:
            seg_preds = preds[mask.values]
            seg_actual = actual[mask.values]
            seg_mape = np.mean(np.abs((seg_actual - seg_preds) / seg_actual)) * 100
            seg_mae = np.mean(np.abs(seg_actual - seg_preds))
            print(f"    {name}: n={mask.sum()}, MAPE={seg_mape:.1f}%, MAE=£{seg_mae:,.0f}")


def export_to_chrome(model, feature_cols, output_dir):
    """Export model to Chrome extension format."""
    print("\n" + "=" * 70)
    print("EXPORTING TO CHROME EXTENSION")
    print("=" * 70)

    chrome_dir = Path('chrome-extension/api')
    chrome_dir.mkdir(parents=True, exist_ok=True)

    # Export model to JSON
    model_path = chrome_dir / 'model.json'
    model.save_model(str(model_path.with_suffix('.json')))
    print(f"  Saved model to {model_path}")

    # Export features list
    features_path = chrome_dir / 'features.json'
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved features to {features_path}")

    print(f"\n  Total features: {len(feature_cols)}")
    print(f"  Model trees: {model.n_estimators}")

    return model_path, features_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Skip Optuna tuning')
    parser.add_argument('--export', action='store_true', help='Export to Chrome extension')
    args = parser.parse_args()

    print("=" * 70)
    print("LONDON RENTAL PRICE MODELS V20 - MERGED V16 + V19")
    print("=" * 70)
    print("V16: social housing, postcode adjustment, size anomalies, bathroom signals")
    print("V19: garden squares, prime streets, mews PPSF, type target encoding")

    df = load_and_clean_data()
    df = engineer_features_v20(df)

    feature_cols = get_feature_columns_v20(df)
    X = df[feature_cols].fillna(0)

    # Ensure all columns are numeric (convert booleans)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
        elif X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    y = df['price_pcm']

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")

    results = {}

    # Train default model
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    print("\n[XGBoost_V20_Default]")
    model = build_xgboost()
    result = evaluate_model(model, X, y)
    results['XGBoost_V20_Default'] = result
    print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
    analyze_errors_by_tier(result['actual'], result['predictions'])
    analyze_v20_segments(df, result['predictions'], result['actual'])

    # Optuna tuning
    if not args.quick and HAS_OPTUNA:
        print("\n[XGBoost_V20_Optuna]")
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
            results['XGBoost_V20_Optuna'] = result
            print(f"  R²: {result['R2']:.4f}, MAE: £{result['MAE']:,.0f}, MAPE: {result['MAPE']:.1f}%")
            analyze_errors_by_tier(result['actual'], result['predictions'])
            analyze_v20_segments(df, result['predictions'], result['actual'])

    # Best model
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
    print("TOP 30 FEATURE IMPORTANCE")
    print("-" * 70)

    final_model = build_xgboost()
    final_model.fit(X, np.log1p(y))

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(30).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:<35} {row['importance']:.3f} {bar}")

    # Save model
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / 'rental_model_v20.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open(OUTPUT_DIR / 'rental_model_v20_features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"\nSaved model to {OUTPUT_DIR / 'rental_model_v20.pkl'}")

    # Export to Chrome if requested
    if args.export:
        export_to_chrome(final_model, feature_cols, OUTPUT_DIR)

    # Version comparison
    print("\n" + "=" * 70)
    print("VERSION COMPARISON")
    print("=" * 70)
    print("V15 baseline:  R²=0.790, MAPE=19.7%")
    print("V16 (new):     R²=0.810, MAPE=18.7%")
    print(f"V20 (merged):  R²={best['R2']:.3f}, MAPE={best['MAPE']:.1f}%")
    print(f"Improvement:   R² +{(best['R2'] - 0.790)*100:.1f}%, MAPE -{19.7 - best['MAPE']:.1f}%")

    return results, final_model, feature_cols


if __name__ == '__main__':
    results, model, features = main()
