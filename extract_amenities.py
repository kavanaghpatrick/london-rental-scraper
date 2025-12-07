"""
Extract amenities from property summaries and save to database.

Extracts:
- Balcony, terrace, roof terrace, garden
- Air conditioning
- High ceilings, floor-to-ceiling windows
- Floor level
- Porter/concierge
- Parking, gym, pool
- Furnished status

Usage:
    python extract_amenities.py
    python extract_amenities.py --export  # Also export to CSV
"""

import sqlite3
import pandas as pd
import re
import json
import argparse
from datetime import datetime


def extract_amenities(text):
    """Extract amenities from summary text."""
    if not text or pd.isna(text):
        return {}

    text = text.lower()

    amenities = {}

    # Balcony/Outdoor
    amenities['has_balcony'] = bool(re.search(r'\bbalcon', text))
    amenities['has_terrace'] = bool(re.search(r'\bterrace\b', text))
    amenities['has_roof_terrace'] = bool(re.search(r'roof\s*terrace', text))
    amenities['has_garden'] = bool(re.search(r'\bgarden\b', text))
    amenities['has_outdoor_space'] = any([
        amenities['has_balcony'], amenities['has_terrace'],
        amenities['has_roof_terrace'], amenities['has_garden']
    ])

    # Climate control
    amenities['has_ac'] = bool(re.search(r'air[\s-]*condition|a/c\b|\bac\b|aircon|air con', text))
    amenities['has_underfloor_heating'] = bool(re.search(r'underfloor\s*heat', text))

    # Ceilings/Windows
    amenities['has_high_ceilings'] = bool(re.search(r'high\s*ceiling|double[\s-]*height|lofty\s*ceiling|tall\s*ceiling', text))
    amenities['has_floor_to_ceiling'] = bool(re.search(r'floor[\s-]*to[\s-]*ceiling|full[\s-]*height\s*window', text))

    # Building features
    amenities['has_lift'] = bool(re.search(r'\blift\b|\belevator\b|\bwith lift\b', text))
    amenities['has_porter'] = bool(re.search(r'\bporter|concierge|24[\s-]*hour|24hr\s*security', text))
    amenities['has_gym'] = bool(re.search(r'\bgym\b|fitness|leisure\s*suite', text))
    amenities['has_pool'] = bool(re.search(r'\bpool\b|swimming', text))

    # Parking
    amenities['has_parking'] = bool(re.search(r'\bparking\b|\bgarage\b|car\s*space', text))

    # Quality indicators
    amenities['is_luxury'] = bool(re.search(r'\bluxur|premium|exclusive|prestigious|stunning|magnificent', text))
    amenities['is_refurbished'] = bool(re.search(r'refurbish|renovate|newly\s*decorated|recently\s*refurb', text))
    amenities['is_period'] = bool(re.search(r'period\s*(property|build|feature)', text))
    amenities['is_modern'] = bool(re.search(r'\bmodern\b', text))

    # Floor level
    floor = None
    match = re.search(r'(\d+)(?:st|nd|rd|th)\s*floor', text)
    if match:
        floor = int(match.group(1))
    elif 'ground floor' in text:
        floor = 0
    elif re.search(r'basement|lower\s*ground', text):
        floor = -1
    elif re.search(r'penthouse|top\s*floor', text):
        floor = 99
    amenities['floor_level'] = floor

    # Furnished status
    if 'unfurnished' in text:
        amenities['furnished'] = 'unfurnished'
    elif re.search(r'\bfurnished\b', text) and 'unfurnished' not in text:
        amenities['furnished'] = 'furnished'
    elif 'part furnished' in text or 'part-furnished' in text:
        amenities['furnished'] = 'part'
    else:
        amenities['furnished'] = None

    return amenities


def main():
    parser = argparse.ArgumentParser(description='Extract amenities from property summaries')
    parser.add_argument('--export', action='store_true', help='Export results to CSV')
    args = parser.parse_args()

    print("=" * 70)
    print("PROPERTY AMENITY EXTRACTION")
    print("=" * 70)

    conn = sqlite3.connect('output/rentals.db')

    # Load properties with summaries
    df = pd.read_sql("""
        SELECT id, source, url, price_pcm, bedrooms, bathrooms, size_sqft,
               summary, area, postcode, property_type
        FROM listings_deduped
        WHERE size_sqft > 0 AND summary IS NOT NULL AND summary != ''
    """, conn)

    print(f"\nProperties with summaries: {len(df)}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    # Extract amenities
    amenity_cols = [
        'has_balcony', 'has_terrace', 'has_roof_terrace', 'has_garden', 'has_outdoor_space',
        'has_ac', 'has_underfloor_heating', 'has_high_ceilings', 'has_floor_to_ceiling',
        'has_lift', 'has_porter', 'has_gym', 'has_pool', 'has_parking',
        'is_luxury', 'is_refurbished', 'is_period', 'is_modern',
        'floor_level', 'furnished'
    ]

    for col in amenity_cols:
        df[col] = None

    for idx, row in df.iterrows():
        amenities = extract_amenities(row['summary'])
        for k, v in amenities.items():
            df.at[idx, k] = v

    # Update database with features JSON
    print("\nUpdating database with extracted features...")
    cursor = conn.cursor()

    update_count = 0
    for idx, row in df.iterrows():
        amenities = {col: row[col] for col in amenity_cols if row[col] is not None and row[col] != False}
        if amenities:
            features_json = json.dumps(amenities)
            cursor.execute("UPDATE listings SET features = ? WHERE id = ?", (features_json, row['id']))
            update_count += 1

    conn.commit()
    print(f"Updated {update_count} records with amenity features")

    # Generate report
    print("\n" + "=" * 70)
    print("AMENITY SUMMARY")
    print("=" * 70)

    overall_avg = df['price_pcm'].mean()

    print(f"\n{'Amenity':<25} {'Count':>6} {'%':>6} {'Avg Price':>12} {'Premium':>10}")
    print("-" * 70)

    # Boolean amenities
    bool_cols = [c for c in amenity_cols if c.startswith('has_') or c.startswith('is_')]
    results = []

    for col in bool_cols:
        has_amenity = df[col] == True
        count = has_amenity.sum()
        if count > 0:
            pct = 100 * count / len(df)
            avg_price = df.loc[has_amenity, 'price_pcm'].mean()
            premium = (avg_price - overall_avg) / overall_avg * 100
            results.append((col, count, pct, avg_price, premium))

    results.sort(key=lambda x: x[4], reverse=True)
    for col, count, pct, avg_price, premium in results:
        name = col.replace('has_', '').replace('is_', '')
        print(f"{name:<25} {count:>6} {pct:>5.1f}% £{avg_price:>10,.0f} {premium:>+9.1f}%")

    # KEY AMENITIES REPORT
    print("\n" + "=" * 70)
    print("PROPERTIES WITH KEY AMENITIES")
    print("=" * 70)

    key_amenities = {
        'Air Conditioning': 'has_ac',
        'Pool': 'has_pool',
        'High Ceilings': 'has_high_ceilings',
        'Floor-to-Ceiling Windows': 'has_floor_to_ceiling',
        'Balcony': 'has_balcony',
        'Terrace': 'has_terrace',
        'Porter/Concierge': 'has_porter',
    }

    for name, col in key_amenities.items():
        subset = df[df[col] == True].sort_values('price_pcm', ascending=False)
        print(f"\n[{name.upper()}] ({len(subset)} properties)")

        for _, row in subset.head(5).iterrows():
            beds = int(row['bedrooms']) if pd.notna(row['bedrooms']) else 0
            sqft = int(row['size_sqft']) if pd.notna(row['size_sqft']) else 0
            print(f"  £{row['price_pcm']:,.0f} | {beds}bed/{sqft}sqft | {row['area']}")
            print(f"    {row['url']}")

    # Floor level report
    print("\n" + "=" * 70)
    print("FLOOR LEVEL DISTRIBUTION")
    print("=" * 70)

    floor_data = df[df['floor_level'].notna()].copy()
    print(f"\nProperties with floor info: {len(floor_data)} ({100*len(floor_data)/len(df):.1f}%)")

    if len(floor_data) > 0:
        floor_data['floor_name'] = floor_data['floor_level'].apply(
            lambda x: "Basement" if x == -1 else ("Ground" if x == 0 else
                      ("Penthouse/Top" if x == 99 else f"Floor {int(x)}"))
        )

        for floor_name in ['Ground', 'Floor 2', 'Floor 3', 'Floor 4', 'Floor 5', 'Penthouse/Top']:
            subset = floor_data[floor_data['floor_name'] == floor_name]
            if len(subset) > 0:
                print(f"\n[{floor_name.upper()}] ({len(subset)} properties)")
                for _, row in subset.head(3).iterrows():
                    beds = int(row['bedrooms']) if pd.notna(row['bedrooms']) else 0
                    print(f"  £{row['price_pcm']:,.0f} | {beds}bed | {row['area']}")
                    print(f"    {row['url']}")

    # Export to CSV if requested
    if args.export:
        output_file = 'output/amenities_report.csv'
        export_cols = ['url', 'price_pcm', 'bedrooms', 'size_sqft', 'area', 'postcode'] + amenity_cols
        df[export_cols].to_csv(output_file, index=False)
        print(f"\nExported to {output_file}")

    conn.close()
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    return df


if __name__ == '__main__':
    df = main()
