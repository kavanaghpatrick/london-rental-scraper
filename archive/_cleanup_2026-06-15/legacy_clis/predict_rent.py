#!/usr/bin/env python3
"""
Rental Price Predictor - V14 Model

Predicts monthly rent for London properties using the V14 model.
Model: XGBoost with log target, MAPE ~24%, Median APE ~18%

Usage:
    # Interactive mode
    python scripts/predict_rent.py

    # Command line
    python scripts/predict_rent.py --size 1200 --beds 2 --baths 2 --postcode SW3

    # With amenities
    python scripts/predict_rent.py --size 1500 --beds 2 --baths 2 --postcode W1 \
        --has-porter --has-gym --is-period

Known Limitations:
    - Porter/gym features have limited training samples (69 properties)
    - Location features (Knightsbridge, Mayfair) have strong signal
    - Best for prime London (SW, W postcodes)
"""

import pickle
import numpy as np
import argparse
import re
from pathlib import Path

# Load model and info
MODEL_DIR = Path(__file__).parent.parent / 'output'

def load_model():
    with open(MODEL_DIR / 'rental_model_v14.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(MODEL_DIR / 'rental_model_v14_info.pkl', 'rb') as f:
        info = pickle.load(f)
    with open(MODEL_DIR / 'rental_model_v14_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, info, features


def get_postcode_ppsf(postcode, info):
    """Get blended PPSF for a postcode using trained encodings."""
    # Extract district (e.g., SW3, W1, NW8)
    match = re.match(r'^([A-Z]+\d+)', postcode.upper())
    district = match.group(1) if match else None

    # Extract area (e.g., SW, W, NW)
    match = re.match(r'^([A-Z]+)', postcode.upper())
    area = match.group(1) if match else None

    district_data = info.get('district_ppsf', {})
    area_data = info.get('area_ppsf', {})
    global_median = info['global_ppsf_median']

    # Try district first
    if district and district in district_data:
        d_info = district_data[district]
        if isinstance(d_info, dict):
            d_ppsf = d_info.get('district_ppsf', global_median)
            d_count = d_info.get('district_count', 0)
        else:
            d_ppsf = d_info
            d_count = 10  # Assume sufficient samples

        # Blend with area if few samples
        weight = min(d_count / 10, 1.0)
        a_ppsf = area_data.get(area, global_median) if area else global_median
        return weight * d_ppsf + (1 - weight) * a_ppsf

    # Try area
    if area and area in area_data:
        return area_data[area]

    return global_median


def predict_rent(
    size_sqft: float,
    bedrooms: int,
    bathrooms: int,
    postcode: str,
    source: str = 'rightmove',
    floor_count: int = 1,
    # Floor features
    has_basement: bool = False,
    has_lower_ground: bool = False,
    has_ground: bool = False,
    has_mezzanine: bool = False,
    has_first_floor: bool = False,
    has_second_floor: bool = False,
    has_third_floor: bool = False,
    has_fourth_plus: bool = False,
    has_roof_terrace: bool = False,
    # Amenities
    has_garden: bool = False,
    has_terrace: bool = False,
    has_balcony: bool = False,
    has_parking: bool = False,
    has_porter: bool = False,
    has_concierge: bool = False,
    has_gym: bool = False,
    has_pool: bool = False,
    has_lift: bool = False,
    has_cinema: bool = False,
    has_ensuite: bool = False,
    has_ac: bool = False,
    has_wood_floors: bool = False,
    # Property type
    is_period: bool = False,
    is_penthouse: bool = False,
    is_lateral: bool = False,
    # Address features
    is_square: bool = False,
    is_place: bool = False,
    is_crescent: bool = False,
    is_gate: bool = False,
    near_hyde_park: bool = False,
    # Location
    is_knightsbridge: bool = False,
    is_mayfair: bool = False,
    is_belgravia: bool = False,
    is_chelsea: bool = False,
    is_kensington: bool = False,
    is_notting_hill: bool = False,
) -> dict:
    """
    Predict monthly rent for a property.

    Returns dict with:
        - predicted_rent: Predicted monthly rent in £
        - confidence_range: (low, high) range based on model error
        - ppsf: Predicted price per square foot
        - comparisons: How this compares to area averages
    """
    model, info, feature_names = load_model()

    # Get encodings
    postcode_ppsf = get_postcode_ppsf(postcode, info)
    source_ppsf = info.get('source_ppsf', {}).get(source, info['global_ppsf_median'])

    # Check if prime
    district = re.match(r'^([A-Z]+\d+)', postcode.upper())
    district = district.group(1) if district else ''
    is_prime = 1 if district in info['prime_districts'] else 0

    # Build feature vector
    features = {
        # Core
        'size_sqft': size_sqft,
        'log_size': np.log1p(size_sqft),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_per_bed': size_sqft / max(bedrooms, 1),
        'floor_count': floor_count,

        # Encodings
        'postcode_ppsf': postcode_ppsf,
        'source_ppsf': source_ppsf,
        'is_prime': is_prime,

        # Interactions
        'size_x_ppsf': np.log1p(size_sqft) * postcode_ppsf,
        'size_x_prime': size_sqft * is_prime,

        # Floors
        'has_basement': int(has_basement),
        'has_lower_ground': int(has_lower_ground),
        'has_ground': int(has_ground),
        'has_mezzanine': int(has_mezzanine),
        'has_first_floor': int(has_first_floor),
        'has_second_floor': int(has_second_floor),
        'has_third_floor': int(has_third_floor),
        'has_fourth_plus': int(has_fourth_plus),
        'has_roof_terrace': int(has_roof_terrace),

        # Amenities
        'has_garden': int(has_garden),
        'has_terrace': int(has_terrace),
        'has_balcony': int(has_balcony),
        'has_parking': int(has_parking),
        'has_porter': int(has_porter),
        'has_concierge': int(has_concierge),
        'has_gym': int(has_gym),
        'has_pool': int(has_pool),
        'has_lift': int(has_lift),
        'has_cinema': int(has_cinema),
        'has_ensuite': int(has_ensuite),
        'has_ac': int(has_ac),
        'has_wood_floors': int(has_wood_floors),

        # Property type
        'is_period': int(is_period),
        'is_penthouse': int(is_penthouse),
        'is_lateral': int(is_lateral),

        # Address
        'is_square': int(is_square),
        'is_place': int(is_place),
        'is_crescent': int(is_crescent),
        'is_gate': int(is_gate),
        'near_hyde_park': int(near_hyde_park),

        # Location
        'is_knightsbridge': int(is_knightsbridge),
        'is_mayfair': int(is_mayfair),
        'is_belgravia': int(is_belgravia),
        'is_chelsea': int(is_chelsea),
        'is_kensington': int(is_kensington),
        'is_notting_hill': int(is_notting_hill),
    }

    # Luxury score
    features['luxury_score'] = sum([
        features['has_pool'], features['has_cinema'],
        features['has_gym'], features['has_porter'], features['has_concierge']
    ])

    # Location score
    features['location_score'] = sum([
        features['is_knightsbridge'], features['is_mayfair'], features['is_belgravia'],
        features['is_chelsea'], features['is_kensington'], features['is_notting_hill'],
        features['is_prime'], features['near_hyde_park'], features['is_square']
    ])

    # Create feature array in correct order
    X = np.array([[features.get(f, 0) for f in feature_names]])

    # Predict (model predicts log1p(price))
    log_pred = model.predict(X)[0]
    predicted_rent = np.expm1(log_pred)

    # Confidence range (based on ~18% median APE)
    low = predicted_rent * 0.82
    high = predicted_rent * 1.18

    # Calculate implied PPSF
    ppsf = predicted_rent / size_sqft

    return {
        'predicted_rent': round(predicted_rent),
        'confidence_range': (round(low), round(high)),
        'ppsf': round(ppsf, 2),
        'postcode_avg_ppsf': round(postcode_ppsf, 2),
        'premium_vs_area': round((ppsf / postcode_ppsf - 1) * 100, 1),
    }


def interactive_mode():
    """Interactive prediction mode."""
    print("=" * 60)
    print("LONDON RENTAL PRICE PREDICTOR (V14)")
    print("=" * 60)
    print()

    # Required inputs
    size_sqft = float(input("Size (sqft): "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    postcode = input("Postcode (e.g., SW3, W1 4AB): ").strip()

    print("\nOptional features (press Enter to skip):")

    # Optional amenities
    amenities = {}
    amenity_prompts = [
        ('has_porter', 'Has porter/doorman? (y/n): '),
        ('has_gym', 'Has gym? (y/n): '),
        ('has_pool', 'Has pool? (y/n): '),
        ('has_terrace', 'Has terrace? (y/n): '),
        ('has_garden', 'Has garden? (y/n): '),
        ('has_balcony', 'Has balcony? (y/n): '),
        ('has_parking', 'Has parking? (y/n): '),
        ('has_lift', 'Has lift? (y/n): '),
        ('is_period', 'Period property? (y/n): '),
        ('is_penthouse', 'Penthouse? (y/n): '),
    ]

    for key, prompt in amenity_prompts:
        resp = input(prompt).strip().lower()
        amenities[key] = resp in ('y', 'yes', '1', 'true')

    # Location detection from postcode
    postcode_upper = postcode.upper()
    location_flags = {
        'is_knightsbridge': 'knightsbridge' in postcode_upper.lower() or postcode_upper.startswith('SW1X') or postcode_upper.startswith('SW7'),
        'is_mayfair': postcode_upper.startswith('W1'),
        'is_belgravia': postcode_upper.startswith('SW1'),
        'is_chelsea': postcode_upper.startswith('SW3') or postcode_upper.startswith('SW10'),
        'is_kensington': postcode_upper.startswith('W8') or postcode_upper.startswith('W14'),
        'is_notting_hill': postcode_upper.startswith('W11'),
    }

    # Predict
    result = predict_rent(
        size_sqft=size_sqft,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        postcode=postcode,
        **amenities,
        **location_flags
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n  Predicted Rent:  £{result['predicted_rent']:,} pcm")
    print(f"  Confidence:      £{result['confidence_range'][0]:,} - £{result['confidence_range'][1]:,}")
    print(f"\n  Price/sqft:      £{result['ppsf']}/sqft")
    print(f"  Area average:    £{result['postcode_avg_ppsf']}/sqft")
    print(f"  Premium:         {result['premium_vs_area']:+.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description='Predict London rental prices')
    parser.add_argument('--size', type=float, help='Size in sqft')
    parser.add_argument('--beds', type=int, help='Number of bedrooms')
    parser.add_argument('--baths', type=int, help='Number of bathrooms')
    parser.add_argument('--postcode', type=str, help='Postcode (e.g., SW3, W1)')
    parser.add_argument('--source', type=str, default='rightmove', help='Listing source')

    # Boolean flags
    parser.add_argument('--has-porter', action='store_true')
    parser.add_argument('--has-gym', action='store_true')
    parser.add_argument('--has-pool', action='store_true')
    parser.add_argument('--has-terrace', action='store_true')
    parser.add_argument('--has-garden', action='store_true')
    parser.add_argument('--has-balcony', action='store_true')
    parser.add_argument('--has-parking', action='store_true')
    parser.add_argument('--has-lift', action='store_true')
    parser.add_argument('--has-cinema', action='store_true')
    parser.add_argument('--has-ac', action='store_true')
    parser.add_argument('--has-wood-floors', action='store_true')
    parser.add_argument('--is-period', action='store_true')
    parser.add_argument('--is-penthouse', action='store_true')
    parser.add_argument('--is-lateral', action='store_true')
    parser.add_argument('--near-hyde-park', action='store_true')

    args = parser.parse_args()

    # If no args, run interactive mode
    if not args.size:
        interactive_mode()
        return

    # Command line mode
    if not all([args.size, args.beds, args.baths, args.postcode]):
        print("Error: --size, --beds, --baths, and --postcode are required")
        return

    result = predict_rent(
        size_sqft=args.size,
        bedrooms=args.beds,
        bathrooms=args.baths,
        postcode=args.postcode,
        source=args.source,
        has_porter=args.has_porter,
        has_gym=args.has_gym,
        has_pool=args.has_pool,
        has_terrace=args.has_terrace,
        has_garden=args.has_garden,
        has_balcony=args.has_balcony,
        has_parking=args.has_parking,
        has_lift=args.has_lift,
        has_cinema=args.has_cinema,
        has_ac=args.has_ac,
        has_wood_floors=args.has_wood_floors,
        is_period=args.is_period,
        is_penthouse=args.is_penthouse,
        is_lateral=args.is_lateral,
        near_hyde_park=args.near_hyde_park,
    )

    print(f"\nPredicted Rent: £{result['predicted_rent']:,} pcm")
    print(f"Range: £{result['confidence_range'][0]:,} - £{result['confidence_range'][1]:,}")
    print(f"PPSF: £{result['ppsf']:.2f}/sqft (area avg: £{result['postcode_avg_ppsf']:.2f}/sqft)")


if __name__ == '__main__':
    main()
