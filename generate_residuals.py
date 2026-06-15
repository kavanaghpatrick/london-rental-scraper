"""
Generate predictions and residuals for all properties in the database.
This creates a CSV file for parallel agents to analyze.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import xgboost as xgb
from pathlib import Path

# Load model and features
model = xgb.XGBRegressor()
model.load_model('chrome-extension/api/model.json')

with open('chrome-extension/api/features.json') as f:
    FEATURES = json.load(f)

print(f"Model loaded with {len(FEATURES)} features")

# Constants
PPSF_MIN, PPSF_MAX = 3, 30
SQFT_MIN, SQFT_MAX = 150, 10000
SQFT_PER_BED_MIN = 70
PRIME_POSTCODES = ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW3', 'NW8']
PREMIUM_AGENTS = ['Knight Frank', 'Savills', 'Harrods Estates', 'Sotheby',
                  'Beauchamp Estates', 'Strutt & Parker', 'Chestertons']

GARDEN_SQUARES = [
    'cadogan square', 'belgrave square', 'chester square', 'eaton square',
    'montpelier square', 'brompton square', 'thurloe square', 'lowndes square',
    'trevor square', 'lennox gardens', 'cadogan gardens', 'sloane square',
    'paultons square', 'chelsea square', 'onslow square', 'pelham crescent',
    'egerton crescent', 'egerton gardens', 'ovington square'
]

ULTRA_PRIME_ADDRESSES = [
    'belgrave square', 'chester square', 'eaton square', 'wilton crescent',
    'grosvenor square', 'grosvenor crescent', 'upper grosvenor street',
    'park lane', 'hamilton terrace', 'avenue road', 'bishops avenue'
]

PRIME_STREETS = [
    'cadogan square', 'cadogan place', 'cadogan gardens', 'hans place',
    'lennox gardens', 'pont street', 'sloane street', 'draycott place',
    'draycott avenue', 'eaton place', 'eaton terrace', 'montpelier street',
    'brompton square', 'thurloe square', 'ennismore gardens', 'princes gate',
    'hyde park gate', 'kensington palace gardens', 'palace gardens terrace',
    'campden hill', 'holland park', 'phillimore gardens', 'carlyle square',
    'cheyne walk', 'the boltons', 'tregunter road', 'elm park gardens'
]

HOUSE_TYPES = ['house', 'terraced', 'detached', 'semi-detached', 'town house', 'cottage', 'end of terrace', 'link detached']
FLAT_TYPES = ['flat', 'apartment', 'studio', 'penthouse', 'maisonette', 'duplex', 'ground flat']

TUBE_STATIONS = {
    'South Kensington': (51.4941, -0.1738), 'Sloane Square': (51.4924, -0.1565),
    'Knightsbridge': (51.5015, -0.1607), 'Hyde Park Corner': (51.5027, -0.1527),
    'Victoria': (51.4965, -0.1447),
}

POSTCODE_CENTROIDS = {
    'SW1': (51.4970, -0.1400), 'SW3': (51.4900, -0.1680), 'SW5': (51.4920, -0.1940),
    'SW7': (51.4950, -0.1780), 'SW10': (51.4830, -0.1820), 'SW11': (51.4650, -0.1650),
    'W1': (51.5150, -0.1450), 'W2': (51.5150, -0.1780), 'W8': (51.5010, -0.1920),
    'W11': (51.5150, -0.2050), 'NW3': (51.5550, -0.1780), 'NW8': (51.5330, -0.1750),
}

CITY_CENTER = (51.5074, -0.1278)

AMENITY_FEATURES = [
    'has_balcony', 'has_terrace', 'has_garden', 'has_porter',
    'has_gym', 'has_pool', 'has_parking', 'has_lift', 'has_ac',
    'has_furnished', 'has_high_ceilings', 'has_view',
    'has_modern', 'has_period', 'has_roof_terrace'
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_tube_dist(lat, lon):
    if pd.isna(lat) or lat == 0:
        return 1.0
    return min(haversine(lat, lon, s[0], s[1]) for s in TUBE_STATIONS.values())


def get_center_dist(lat, lon):
    if pd.isna(lat) or lat == 0:
        return 3.0
    return haversine(lat, lon, CITY_CENTER[0], CITY_CENTER[1])


def parse_amenities(features_str, desc=''):
    amenities = {f: 0 for f in AMENITY_FEATURES}
    text = (str(features_str or '') + ' ' + str(desc or '')).lower()

    amenities['has_balcony'] = int('balcony' in text)
    amenities['has_terrace'] = int('terrace' in text and 'roof terrace' not in text)
    amenities['has_roof_terrace'] = int('roof terrace' in text)
    amenities['has_garden'] = int('garden' in text)
    amenities['has_porter'] = int('porter' in text or 'concierge' in text)
    amenities['has_gym'] = int('gym' in text)
    amenities['has_pool'] = int('pool' in text or 'swimming' in text)
    amenities['has_parking'] = int('parking' in text or 'garage' in text)
    amenities['has_lift'] = int('lift' in text or 'elevator' in text)
    amenities['has_ac'] = int('air con' in text or 'a/c' in text)
    amenities['has_high_ceilings'] = int('high ceiling' in text)
    amenities['has_view'] = int('view' in text)
    amenities['has_modern'] = int('modern' in text or 'contemporary' in text)
    amenities['has_period'] = int('period' in text or 'victorian' in text)
    amenities['has_furnished'] = int('furnished' in text and 'unfurnished' not in text)

    return amenities


def build_features_for_row(row, global_stats):
    """Build feature vector for a single row."""
    X = {f: 0.0 for f in FEATURES}

    # Core
    beds = row['bedrooms'] or 1
    baths = row['bathrooms'] or 1
    sqft = row['size_sqft'] or 500

    X['bedrooms'] = beds
    X['bathrooms'] = baths
    X['size_sqft'] = sqft
    X['size_per_bed'] = sqft / max(beds, 0.5)
    X['bath_ratio'] = baths / max(beds, 0.5)
    X['bed_bath_interaction'] = beds * baths
    X['log_sqft'] = np.log1p(sqft)
    X['sqrt_sqft'] = np.sqrt(sqft)
    X['size_squared'] = sqft**2 / 100000
    X['beds_squared'] = beds**2
    X['size_bin'] = global_stats['size_bins'].get(sqft, 2)

    # Location
    lat, lon = row['latitude'], row['longitude']
    pc = str(row.get('postcode_normalized') or row.get('postcode') or 'SW3')
    pc_district = pc[:4] if len(pc) >= 4 else pc

    X['tube_distance_km'] = get_tube_dist(lat, lon)
    X['log_tube_distance'] = np.log1p(X['tube_distance_km'])
    X['center_distance_km'] = get_center_dist(lat, lon)
    X['log_center_distance'] = np.log1p(X['center_distance_km'])
    X['center_distance_inv'] = 1 / (1 + X['center_distance_km'])

    is_prime = int(any(pc_district.startswith(p) for p in PRIME_POSTCODES))
    X['is_prime_postcode'] = is_prime
    X['postcode_freq'] = global_stats['pc_freq'].get(pc_district, 0.01)
    X['postcode_area_freq'] = global_stats['area_freq'].get(pc_district[:2], 0.05)

    # Size x location
    X['size_x_central'] = sqft * X['center_distance_inv'] / 100
    X['size_x_prime'] = sqft * is_prime / 1000
    X['beds_x_central'] = beds * X['center_distance_inv']

    # Floor
    X['floor_count'] = row.get('floor_count') or 0
    X['is_multi_floor'] = int(X['floor_count'] >= 2)
    X['floor_size_interaction'] = X['floor_count'] * sqft / 1000
    X['has_basement'] = row.get('has_basement') or 0
    X['has_ground'] = row.get('has_ground') or 0
    X['has_first_floor'] = row.get('has_first_floor') or 0
    X['has_second_floor'] = row.get('has_second_floor') or 0
    X['has_third_floor'] = row.get('has_third_floor') or 0
    X['has_fourth_plus'] = row.get('has_fourth_plus') or 0

    # Agent
    agent = str(row.get('agent_brand') or '')
    X['is_premium_agent'] = int(any(a.lower() in agent.lower() for a in PREMIUM_AGENTS))
    X['premium_agent_size'] = X['is_premium_agent'] * X['log_sqft']

    source_map = {'savills': 4, 'knightfrank': 4, 'chestertons': 3, 'foxtons': 2, 'rightmove': 1}
    X['source_quality'] = source_map.get(str(row.get('source') or '').lower(), 2)

    # Amenities
    amenities = parse_amenities(row.get('features'), row.get('description'))
    for f in AMENITY_FEATURES:
        X[f] = amenities.get(f, 0)

    X['amenity_score'] = sum(amenities.values())
    X['premium_amenity_count'] = amenities['has_pool'] + amenities['has_porter'] + amenities['has_gym'] + amenities['has_ac']
    X['has_outdoor_space'] = int(amenities['has_balcony'] or amenities['has_terrace'] or amenities['has_garden'] or amenities['has_roof_terrace'])
    X['amenity_x_central'] = X['amenity_score'] * X['center_distance_inv']
    X['outdoor_x_prime'] = X['has_outdoor_space'] * is_prime

    # Let type
    let_type = str(row.get('let_type') or '').lower()
    X['is_short_let'] = int('short' in let_type)
    X['is_long_let'] = int('long' in let_type)
    X['short_let_x_central'] = X['is_short_let'] * X['center_distance_inv']
    X['short_let_size'] = X['is_short_let'] * X['log_sqft']

    # Property type
    prop_type = str(row.get('property_type_std') or row.get('property_type') or 'flat').lower()
    type_map = {'studio': 0, 'flat': 1, 'apartment': 1, 'maisonette': 2, 'house': 3, 'penthouse': 4}
    X['property_type_num'] = type_map.get(prop_type, 1)

    # V16 Address premium
    addr = str(row.get('address') or '').lower()
    X['is_garden_square'] = int(any(sq in addr for sq in GARDEN_SQUARES))
    X['is_ultra_prime_address'] = int(any(a in addr for a in ULTRA_PRIME_ADDRESSES))
    X['is_prime_street'] = int(any(st in addr for st in PRIME_STREETS))
    X['garden_square_size'] = X['is_garden_square'] * X['log_sqft']
    X['ultra_prime_size'] = X['is_ultra_prime_address'] * sqft / 1000
    X['prime_street_size'] = X['is_prime_street'] * X['log_sqft']
    X['address_prestige'] = X['is_ultra_prime_address'] * 3 + X['is_garden_square'] * 2 + X['is_prime_street']
    X['prestige_x_size'] = X['address_prestige'] * X['log_sqft']

    # V19 Mews
    is_mews = int('mews' in prop_type or 'mews' in addr)
    X['is_mews'] = is_mews
    X['mews_size_interaction'] = is_mews * X['log_sqft']
    X['mews_x_prime'] = is_mews * is_prime

    # V17 Property type
    is_house = int(not is_mews and any(h in prop_type for h in HOUSE_TYPES))
    is_flat = int(any(f in prop_type for f in FLAT_TYPES))

    X['is_house'] = is_house
    X['is_flat'] = is_flat
    X['house_size_interaction'] = is_house * X['log_sqft']
    X['flat_size_interaction'] = is_flat * X['log_sqft']
    X['is_large_house'] = int(is_house and sqft > 2000)
    X['large_house_size'] = X['is_large_house'] * sqft / 1000

    # Type PPSF
    if is_mews:
        type_ppsf = global_stats['mews_ppsf']
        pc_type_ppsf = global_stats['pc_mews_ppsf'].get(pc_district, type_ppsf)
    elif is_house:
        type_ppsf = global_stats['house_ppsf']
        pc_type_ppsf = global_stats['pc_house_ppsf'].get(pc_district, type_ppsf)
    elif is_flat:
        type_ppsf = global_stats['flat_ppsf']
        pc_type_ppsf = global_stats['pc_flat_ppsf'].get(pc_district, type_ppsf)
    else:
        type_ppsf = global_stats['overall_ppsf']
        pc_type_ppsf = type_ppsf

    X['type_ppsf_target'] = type_ppsf
    X['log_type_expected_price'] = np.log1p(type_ppsf * sqft)
    X['pc_type_ppsf'] = pc_type_ppsf
    X['log_pc_type_expected_price'] = np.log1p(pc_type_ppsf * sqft)

    # V18 Furnished
    desc = str(row.get('description') or '').lower()
    X['is_furnished_explicit'] = int('furnished' in desc and 'unfurnished' not in desc and 'part' not in desc)
    X['is_unfurnished'] = int('unfurnished' in desc)
    X['is_part_furnished'] = int('part furnished' in desc or 'part-furnished' in desc)
    X['furnished_x_prime'] = X['is_furnished_explicit'] * is_prime
    X['furnished_x_central'] = X['is_furnished_explicit'] * X['center_distance_inv']
    X['unfurnished_discount'] = X['is_unfurnished'] * X['log_sqft']

    # One-hot postcodes
    for f in FEATURES:
        if f.startswith('pc_') and f != 'pc_type_ppsf':
            pc_name = f.replace('pc_', '')
            X[f] = int(pc_district == pc_name)

    # One-hot property types
    for f in FEATURES:
        if f.startswith('type_') and f not in ['type_ppsf_target']:
            type_name = f.replace('type_', '')
            X[f] = int(type_name in prop_type)

    return X


def main():
    # Load all data
    query = """
        SELECT
            id, address, postcode, postcode_normalized, area,
            bedrooms, bathrooms, size_sqft, price_pcm,
            property_type, property_type_std, let_type,
            latitude, longitude, features, description, source,
            agent_brand, floor_count, has_roof_terrace, has_basement,
            has_ground, has_first_floor, has_second_floor,
            has_third_floor, has_fourth_plus
        FROM listings
        WHERE is_active = 1
        AND size_sqft > 0
        AND price_pcm > 0
        AND bedrooms > 0
        AND (is_short_let = 0 OR is_short_let IS NULL)
    """

    conn = sqlite3.connect('output/rentals.db')
    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Loaded {len(df)} listings")

    # Apply quality filters
    df['ppsf'] = df['price_pcm'] / df['size_sqft']
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)

    mask = (
        (sqft_per_bed >= SQFT_PER_BED_MIN) &
        (df['price_pcm'] <= 100000) &
        (df['size_sqft'] >= SQFT_MIN) &
        (df['size_sqft'] <= SQFT_MAX) &
        (df['ppsf'] >= PPSF_MIN) &
        (df['ppsf'] <= PPSF_MAX) &
        (df['price_pcm'] >= 500)
    )
    df = df[mask].copy()
    print(f"After quality filters: {len(df)} listings")

    # Compute global stats for target encoding
    df['postcode_district'] = df['postcode_normalized'].fillna(df['postcode']).str.extract(r'^([A-Z]+\d+[A-Z]?)', expand=False)
    df['postcode_area'] = df['postcode_district'].str.extract(r'^([A-Z]+)', expand=False)

    # Property type detection
    df['prop_type_lower'] = df['property_type_std'].fillna(df['property_type']).fillna('flat').str.lower()
    df['addr_lower'] = df['address'].fillna('').str.lower()

    df['is_mews'] = ((df['prop_type_lower'].str.contains('mews')) | (df['addr_lower'].str.contains('mews'))).astype(int)
    df['is_house'] = ((~df['is_mews'].astype(bool)) & (df['prop_type_lower'].str.contains('|'.join(HOUSE_TYPES)))).astype(int)
    df['is_flat'] = df['prop_type_lower'].str.contains('|'.join(FLAT_TYPES)).astype(int)

    global_stats = {
        'overall_ppsf': df['ppsf'].median(),
        'house_ppsf': df[df['is_house'] == 1]['ppsf'].median() if df['is_house'].sum() > 0 else 5.5,
        'flat_ppsf': df[df['is_flat'] == 1]['ppsf'].median() if df['is_flat'].sum() > 0 else 5.3,
        'mews_ppsf': df[df['is_mews'] == 1]['ppsf'].median() if df['is_mews'].sum() > 0 else 5.4,
        'pc_house_ppsf': df[df['is_house'] == 1].groupby('postcode_district')['ppsf'].median().to_dict(),
        'pc_flat_ppsf': df[df['is_flat'] == 1].groupby('postcode_district')['ppsf'].median().to_dict(),
        'pc_mews_ppsf': df[df['is_mews'] == 1].groupby('postcode_district')['ppsf'].median().to_dict(),
        'pc_freq': (df['postcode_district'].value_counts() / len(df)).to_dict(),
        'area_freq': (df['postcode_area'].value_counts() / len(df)).to_dict(),
        'size_bins': {},
    }

    # Compute size bins
    size_bins = pd.qcut(df['size_sqft'], q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
    for sqft, bin_val in zip(df['size_sqft'], size_bins):
        global_stats['size_bins'][sqft] = float(bin_val) if pd.notna(bin_val) else 2.0

    print(f"\nGlobal stats:")
    print(f"  Overall PPSF: £{global_stats['overall_ppsf']:.2f}")
    print(f"  House PPSF: £{global_stats['house_ppsf']:.2f}")
    print(f"  Flat PPSF: £{global_stats['flat_ppsf']:.2f}")
    print(f"  Mews PPSF: £{global_stats['mews_ppsf']:.2f}")

    # Generate predictions
    print("\nGenerating predictions...")
    results = []

    for idx, row in df.iterrows():
        X = build_features_for_row(row, global_stats)
        X_arr = np.array([[X.get(f, 0) for f in FEATURES]], dtype=np.float32)

        log_pred = model.predict(X_arr)[0]
        pred = np.expm1(log_pred)

        actual = row['price_pcm']
        residual = actual - pred
        pct_error = (pred - actual) / actual * 100
        abs_pct_error = abs(pct_error)

        results.append({
            'id': row['id'],
            'address': row['address'],
            'postcode': row['postcode'],
            'postcode_district': row['postcode_district'],
            'bedrooms': row['bedrooms'],
            'bathrooms': row['bathrooms'],
            'size_sqft': row['size_sqft'],
            'price_pcm': actual,
            'predicted': pred,
            'residual': residual,
            'pct_error': pct_error,
            'abs_pct_error': abs_pct_error,
            'ppsf': row['ppsf'],
            'property_type': row['property_type_std'] or row['property_type'],
            'is_mews': row['is_mews'],
            'is_house': row['is_house'],
            'is_flat': row['is_flat'],
            'source': row['source'],
            'agent_brand': row['agent_brand'],
            'description': row['description'][:500] if row['description'] else '',
        })

    results_df = pd.DataFrame(results)

    # Summary stats
    print(f"\n{'='*60}")
    print("RESIDUAL SUMMARY")
    print('='*60)
    print(f"Total properties: {len(results_df)}")
    print(f"Mean absolute error: £{results_df['residual'].abs().mean():,.0f}")
    print(f"Median absolute error: £{results_df['residual'].abs().median():,.0f}")
    print(f"Mean % error: {results_df['pct_error'].mean():.1f}%")
    print(f"Mean absolute % error (MAPE): {results_df['abs_pct_error'].mean():.1f}%")

    print(f"\nOverestimates (pred > actual): {(results_df['pct_error'] > 0).sum()} ({100*(results_df['pct_error'] > 0).mean():.1f}%)")
    print(f"Underestimates (pred < actual): {(results_df['pct_error'] < 0).sum()} ({100*(results_df['pct_error'] < 0).mean():.1f}%)")

    # Worst predictions
    print(f"\n{'='*60}")
    print("WORST OVERESTIMATES (model predicts too high)")
    print('='*60)
    worst_over = results_df.nlargest(20, 'pct_error')
    for _, r in worst_over.iterrows():
        print(f"  {r['pct_error']:+.0f}%: £{r['price_pcm']:,.0f} actual vs £{r['predicted']:,.0f} pred | {r['bedrooms']}bed {r['size_sqft']}sqft {r['property_type']} | {r['address'][:50]}")

    print(f"\n{'='*60}")
    print("WORST UNDERESTIMATES (model predicts too low)")
    print('='*60)
    worst_under = results_df.nsmallest(20, 'pct_error')
    for _, r in worst_under.iterrows():
        print(f"  {r['pct_error']:+.0f}%: £{r['price_pcm']:,.0f} actual vs £{r['predicted']:,.0f} pred | {r['bedrooms']}bed {r['size_sqft']}sqft {r['property_type']} | {r['address'][:50]}")

    # Save results
    output_path = 'output/residuals_analysis.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} predictions to {output_path}")

    return results_df


if __name__ == '__main__':
    main()
