"""
Rent Fair Value API - /api/valuate
Serverless function for Vercel deployment
"""

import json
import os
import re
import pickle
from http.server import BaseHTTPRequestHandler
import numpy as np
import pandas as pd

# ============================================
# CONFIGURATION
# ============================================

API_KEY = os.environ.get('RFV_API_KEY', 'rfv-mvp-key-2024')

# Load size lookup table
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, 'size_lookup.json')) as f:
    SIZE_LOOKUP = json.load(f)

# Load model and feature list
MODEL_PATH = os.path.join(SCRIPT_DIR, 'rental_model_v15.pkl')
FEATURES_PATH = os.path.join(SCRIPT_DIR, 'rental_model_v15_features.pkl')

with open(MODEL_PATH, 'rb') as f:
    MODEL = pickle.load(f)
with open(FEATURES_PATH, 'rb') as f:
    FEATURE_COLS = pickle.load(f)

# ============================================
# CONSTANTS (from rental_price_models_v15.py)
# ============================================

AMENITY_FEATURES = [
    'has_balcony', 'has_terrace', 'has_garden', 'has_porter',
    'has_gym', 'has_pool', 'has_parking', 'has_lift', 'has_ac',
    'has_furnished', 'has_high_ceilings', 'has_view',
    'has_modern', 'has_period', 'has_roof_terrace'
]

PRIME_POSTCODES = ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW3', 'NW8']

PREMIUM_AGENTS = ['Knight Frank', 'Savills', 'Harrods Estates', 'Sotheby',
                  'Beauchamp Estates', 'Strutt & Parker', 'Chestertons']

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

# Property type columns expected by model
PROPERTY_TYPE_COLS = [
    'type_apartment', 'type_detached', 'type_end of terrace', 'type_flat',
    'type_ground flat', 'type_house', 'type_house boat', 'type_house share',
    'type_link detached house', 'type_long let', 'type_maisonette', 'type_mews',
    'type_not specified', 'type_penthouse', 'type_serviced apartments',
    'type_short let', 'type_studio', 'type_terraced', 'type_town house'
]

# Top postcode columns expected by model
POSTCODE_COLS = [
    'pc_SW3', 'pc_SW7', 'pc_W8', 'pc_W2', 'pc_SW5', 'pc_SW11', 'pc_SW10',
    'pc_NW8', 'pc_W11', 'pc_SW1X', 'pc_NW3', 'pc_SW1W', 'pc_W14', 'pc_NW1', 'pc_W10'
]


# ============================================
# HELPER FUNCTIONS
# ============================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_nearest_tube_distance(lat, lon):
    """Get distance to nearest tube station."""
    if lat is None or lon is None or lat == 0 or lon == 0:
        return 2.0  # Default median
    return min(haversine_distance(lat, lon, slat, slon) for slat, slon in TUBE_STATIONS.values())


def get_distance_to_center(lat, lon):
    """Get distance to city center."""
    if lat is None or lon is None or lat == 0 or lon == 0:
        return 5.0  # Default median
    return haversine_distance(lat, lon, CITY_CENTER[0], CITY_CENTER[1])


def parse_price(price_text):
    """Parse price text and convert to PCM."""
    if not price_text:
        return None

    # Extract number
    price_str = re.sub(r'[^\d.]', '', price_text.replace(',', ''))
    if not price_str:
        return None

    price = float(price_str)

    # Check if per week
    text_lower = price_text.lower()
    if 'pw' in text_lower or 'per week' in text_lower or 'weekly' in text_lower:
        price = price * 52 / 12  # Convert to monthly

    return int(price)


def parse_amenities(text):
    """Extract amenity flags from text."""
    amenities = {f: 0 for f in AMENITY_FEATURES}

    if not text:
        return amenities

    text = text.lower()

    amenities['has_balcony'] = int('balcony' in text)
    amenities['has_terrace'] = int('terrace' in text and 'roof terrace' not in text)
    amenities['has_roof_terrace'] = int('roof terrace' in text)
    amenities['has_garden'] = int('garden' in text)
    amenities['has_porter'] = int('porter' in text or 'concierge' in text)
    amenities['has_gym'] = int('gym' in text or 'fitness' in text)
    amenities['has_pool'] = int('pool' in text or 'swimming' in text)
    amenities['has_parking'] = int('parking' in text or 'garage' in text)
    amenities['has_lift'] = int('lift' in text or 'elevator' in text)
    amenities['has_ac'] = int('air con' in text or 'a/c' in text or 'aircon' in text or 'air-con' in text)
    amenities['has_high_ceilings'] = int('high ceiling' in text)
    amenities['has_view'] = int('view' in text)
    amenities['has_modern'] = int('modern' in text or 'contemporary' in text)
    amenities['has_period'] = int('period' in text or 'victorian' in text or 'georgian' in text)
    amenities['has_furnished'] = int('furnished' in text and 'unfurnished' not in text)

    return amenities


def estimate_size(district, beds):
    """Estimate sqft from beds and postcode district."""
    key = f"{district}_{beds}"
    if key in SIZE_LOOKUP:
        return SIZE_LOOKUP[key]

    # Try without trailing letter/number
    base_district = re.match(r'^([A-Z]+\d*)', district)
    if base_district:
        key = f"{base_district.group(1)}_{beds}"
        if key in SIZE_LOOKUP:
            return SIZE_LOOKUP[key]

    # Default heuristic
    return max(200, beds * 400)


def get_coords(postcode_district, lat=None, lon=None):
    """Get coordinates from lat/lon or postcode centroid."""
    if lat and lon and lat != 0 and lon != 0:
        return lat, lon

    # Try exact match
    if postcode_district in POSTCODE_CENTROIDS:
        return POSTCODE_CENTROIDS[postcode_district]

    # Try prefix match
    for key in POSTCODE_CENTROIDS:
        if postcode_district.startswith(key):
            return POSTCODE_CENTROIDS[key]

    # Default to city center
    return CITY_CENTER


def engineer_features(data):
    """Engineer all 93 features from property data."""

    # Extract basic fields
    beds = data.get('bedrooms', 1) or 1
    baths = data.get('bathrooms', 1) or 1
    size_sqft = data.get('size_sqft', 500) or 500
    postcode = data.get('postcode', 'SW3 1AA')
    lat = data.get('latitude')
    lon = data.get('longitude')
    description = data.get('description', '')
    features_text = data.get('features_text', '')
    property_type = data.get('property_type', 'flat')
    agent = data.get('agent', '')
    source = data.get('source', 'rightmove')

    # Parse postcode
    postcode_district = postcode.split()[0] if postcode else 'SW3'
    postcode_area = re.match(r'^([A-Z]+)', postcode_district)
    postcode_area = postcode_area.group(1) if postcode_area else 'SW'

    # Get coordinates
    lat_filled, lon_filled = get_coords(postcode_district, lat, lon)

    # Parse amenities from text
    full_text = f"{description} {features_text}".strip()
    amenities = parse_amenities(full_text)

    # Build feature dict
    row = {}

    # Core features
    row['bedrooms'] = beds
    row['bathrooms'] = baths
    row['size_sqft'] = size_sqft

    beds_adj = beds if beds > 0 else 0.5
    row['size_per_bed'] = size_sqft / beds_adj
    row['bath_ratio'] = baths / beds_adj
    row['bed_bath_interaction'] = beds * baths
    row['log_sqft'] = np.log1p(size_sqft)
    row['sqrt_sqft'] = np.sqrt(size_sqft)
    row['size_squared'] = size_sqft ** 2 / 100000
    row['beds_squared'] = beds ** 2
    row['size_bin'] = 2.0  # Middle bin for single predictions

    # Location features
    tube_dist = get_nearest_tube_distance(lat_filled, lon_filled)
    center_dist = get_distance_to_center(lat_filled, lon_filled)

    row['tube_distance_km'] = tube_dist
    row['log_tube_distance'] = np.log1p(tube_dist)
    row['center_distance_km'] = center_dist
    row['log_center_distance'] = np.log1p(center_dist)
    row['center_distance_inv'] = 1 / (1 + center_dist)

    row['is_prime_postcode'] = int(any(postcode_district.startswith(p) for p in PRIME_POSTCODES))
    row['postcode_freq'] = 0.05  # Default frequency
    row['postcode_area_freq'] = 0.1  # Default frequency

    # Size x location interactions
    row['size_x_central'] = size_sqft * row['center_distance_inv'] / 100
    row['size_x_prime'] = size_sqft * row['is_prime_postcode'] / 1000
    row['beds_x_central'] = beds * row['center_distance_inv']

    # Floor features (default if not provided)
    row['floor_count'] = data.get('floor_count', 1) or 1
    row['is_multi_floor'] = int(row['floor_count'] >= 2)
    row['floor_size_interaction'] = row['floor_count'] * size_sqft / 1000
    row['has_basement'] = data.get('has_basement', 0) or 0
    row['has_ground'] = data.get('has_ground', 0) or 0
    row['has_first_floor'] = data.get('has_first_floor', 0) or 0
    row['has_second_floor'] = data.get('has_second_floor', 0) or 0
    row['has_third_floor'] = data.get('has_third_floor', 0) or 0
    row['has_fourth_plus'] = data.get('has_fourth_plus', 0) or 0

    # Agent/source features
    row['is_premium_agent'] = int(any(pa.lower() in agent.lower() for pa in PREMIUM_AGENTS)) if agent else 0
    row['premium_agent_size'] = row['is_premium_agent'] * row['log_sqft']
    source_map = {'savills': 4, 'knightfrank': 4, 'chestertons': 3, 'foxtons': 2, 'rightmove': 1}
    row['source_quality'] = source_map.get(source.lower(), 2)

    # Amenity features
    row.update(amenities)
    row['amenity_score'] = sum(amenities.values())
    row['premium_amenity_count'] = sum([amenities['has_pool'], amenities['has_porter'],
                                        amenities['has_gym'], amenities['has_ac']])
    row['has_outdoor_space'] = int(amenities['has_balcony'] or amenities['has_terrace'] or
                                   amenities['has_garden'] or amenities['has_roof_terrace'])
    row['amenity_x_central'] = row['amenity_score'] * row['center_distance_inv']
    row['outdoor_x_prime'] = row['has_outdoor_space'] * row['is_prime_postcode']

    # Let type (default to long let)
    row['is_short_let'] = 0
    row['is_long_let'] = 1
    row['short_let_x_central'] = 0
    row['short_let_size'] = 0

    # Property type
    type_map = {'studio': 0, 'flat': 1, 'apartment': 1, 'maisonette': 2, 'house': 3, 'penthouse': 4}
    row['property_type_num'] = type_map.get(property_type.lower(), 1)

    # One-hot property type columns
    for col in PROPERTY_TYPE_COLS:
        type_name = col.replace('type_', '')
        row[col] = int(property_type.lower() == type_name)

    # One-hot postcode columns
    for col in POSTCODE_COLS:
        pc = col.replace('pc_', '')
        row[col] = int(postcode_district == pc)

    return row


# ============================================
# MAIN HANDLER
# ============================================

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key')
        self.end_headers()

    def do_POST(self):
        """Handle valuation request."""
        try:
            # CORS headers
            self.send_header('Access-Control-Allow-Origin', '*')

            # Validate API key
            api_key = self.headers.get('X-API-Key', '')
            if api_key != API_KEY:
                self._send_error(401, 'Invalid API key')
                return

            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)

            # Extract property data
            property_data = data.get('property', {})
            size_sqft_from_ext = data.get('size_sqft')

            # Parse Rightmove property data
            prices = property_data.get('prices', {})
            address = property_data.get('address', {})
            text = property_data.get('text', {})

            price_text = prices.get('primaryPrice', '')
            price_pcm = parse_price(price_text)

            if not price_pcm:
                self._send_error(400, 'Could not parse price')
                return

            beds = property_data.get('bedrooms', 1) or 1
            baths = property_data.get('bathrooms', 1) or 1
            outcode = address.get('outcode', 'SW3')
            incode = address.get('incode', '1AA')
            postcode = f"{outcode} {incode}"
            postcode_district = outcode

            description = text.get('description', '')
            key_features = property_data.get('keyFeatures', [])
            features_text = ' '.join(key_features) if key_features else ''

            property_type = property_data.get('propertySubType', 'flat')
            lat = property_data.get('location', {}).get('latitude')
            lon = property_data.get('location', {}).get('longitude')

            # Get sqft: extension provided → estimate
            if size_sqft_from_ext:
                size_sqft = size_sqft_from_ext
                size_source = 'extension'
            else:
                size_sqft = estimate_size(postcode_district, beds)
                size_source = 'estimated'

            # Build data dict for feature engineering
            prop_data = {
                'bedrooms': beds,
                'bathrooms': baths,
                'size_sqft': size_sqft,
                'postcode': postcode,
                'latitude': lat,
                'longitude': lon,
                'description': description,
                'features_text': features_text,
                'property_type': property_type,
                'agent': '',
                'source': 'rightmove',
            }

            # Engineer features
            features = engineer_features(prop_data)

            # Create DataFrame with correct column order
            df = pd.DataFrame([features])

            # Ensure all feature columns exist
            for col in FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 0

            # Select only the features the model expects
            X = df[FEATURE_COLS]

            # Run prediction
            pred_log = MODEL.predict(X)[0]
            fair_value = int(np.expm1(pred_log))

            # Calculate assessment
            premium_pct = round((price_pcm / fair_value - 1) * 100, 1)

            if premium_pct > 15:
                assessment = 'overpriced'
            elif premium_pct < -10:
                assessment = 'good_deal'
            else:
                assessment = 'fair'

            # Build response
            amenities_detected = [k.replace('has_', '') for k, v in
                                 {k: features[k] for k in AMENITY_FEATURES}.items() if v]

            response = {
                'asking_price': price_pcm,
                'fair_value': fair_value,
                'range_low': int(fair_value * 0.79),
                'range_high': int(fair_value * 1.21),
                'premium_pct': premium_pct,
                'assessment': assessment,
                'size_sqft': size_sqft,
                'size_source': size_source,
                'amenities_detected': amenities_detected,
            }

            self._send_json(200, response)

        except Exception as e:
            print(f"Error: {e}")
            self._send_error(500, str(e))

    def _send_json(self, status, data):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_error(self, status, message):
        """Send error response."""
        self._send_json(status, {'error': message})
