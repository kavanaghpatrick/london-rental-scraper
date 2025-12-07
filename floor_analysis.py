#!/usr/bin/env python3
"""
Floor Level Analysis for Rental Price Prediction

Extracts floor information from text fields and analyzes price impact.
Key hypotheses:
- Higher floors = premium (views, light, less noise)
- Lower ground/basement = discount
- Penthouse = significant premium
- Lift access matters more for higher floors
- Multi-floor (maisonette/duplex) = premium for space, possible discount for stairs
"""

import sqlite3
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'output', 'rentals.db')


def extract_floor_info(row):
    """Extract floor level and characteristics from text fields."""
    text = ' '.join([
        str(row.get('summary', '') or ''),
        str(row.get('description', '') or ''),
        str(row.get('address', '') or '')
    ]).lower()

    result = {
        'floor_level': None,        # Numeric floor level
        'floor_category': None,     # Category (basement, ground, lower, mid, high, penthouse)
        'is_multi_floor': False,    # Spans multiple floors
        'num_floors_span': 1,       # How many floors it spans
        'has_lift': None,           # Lift access mentioned
        'is_penthouse': False,
        'is_basement': False,
        'is_lower_ground': False,
        'is_ground': False,
        'is_top_floor': False,
        'floor_confidence': 'none'  # Confidence in extraction
    }

    # Check for lift
    if re.search(r'\b(lift|elevator)\b', text):
        result['has_lift'] = True
    elif re.search(r'\bno lift\b|walk[- ]?up\b', text):
        result['has_lift'] = False

    # Penthouse detection
    if re.search(r'\bpenthouse\b', text):
        result['is_penthouse'] = True
        result['floor_category'] = 'penthouse'
        result['floor_confidence'] = 'high'

    # Basement detection
    if re.search(r'\bbasement\b', text):
        result['is_basement'] = True
        result['floor_level'] = -1
        result['floor_category'] = 'basement'
        result['floor_confidence'] = 'high'

    # Lower ground detection
    if re.search(r'\blower ground\b', text):
        result['is_lower_ground'] = True
        result['floor_level'] = -0.5  # Between basement and ground
        result['floor_category'] = 'lower_ground'
        result['floor_confidence'] = 'high'

    # Ground floor detection
    if re.search(r'\bground floor\b', text):
        result['is_ground'] = True
        result['floor_level'] = 0
        result['floor_category'] = 'ground'
        result['floor_confidence'] = 'high'

    # Top floor detection
    if re.search(r'\btop floor\b', text):
        result['is_top_floor'] = True
        result['floor_category'] = 'top_floor'
        result['floor_confidence'] = 'medium'

    # Numeric floor extraction
    # Pattern: "5th floor", "fifth floor", "floor 5", "on the 5th"
    ordinal_map = {
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
        'eleventh': 11, 'twelfth': 12
    }

    # Try ordinal words first
    for word, num in ordinal_map.items():
        if re.search(rf'\b{word}\s+floor\b', text):
            result['floor_level'] = num
            result['floor_confidence'] = 'high'
            break

    # Try numeric patterns: "6th floor", "floor 6"
    if result['floor_level'] is None or result['floor_category'] is None:
        match = re.search(r'\b(\d+)(?:st|nd|rd|th)?\s*floor\b', text)
        if match:
            result['floor_level'] = int(match.group(1))
            result['floor_confidence'] = 'high'
        else:
            match = re.search(r'\bfloor\s*(\d+)\b', text)
            if match:
                result['floor_level'] = int(match.group(1))
                result['floor_confidence'] = 'high'

    # Detect multi-floor properties
    multi_patterns = [
        r'\b(two|three|four|five|2|3|4|5)\s+floors?\b',
        r'\bmaisonette\b',
        r'\bduplex\b',
        r'\btriplex\b',
        r'\bover\s+(two|three|four|2|3|4)\s+floors\b',
        r'\bspanning\s+(two|three|2|3)\s+floors\b',
        r'\barranged\s+over\s+(two|three|four|2|3|4)\s+floors\b'
    ]

    for pattern in multi_patterns:
        match = re.search(pattern, text)
        if match:
            result['is_multi_floor'] = True
            # Try to extract number of floors
            num_word = match.group(1) if match.lastindex else None
            if num_word:
                num_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5, '2': 2, '3': 3, '4': 4, '5': 5}
                result['num_floors_span'] = num_map.get(num_word.lower(), 2)
            break

    if 'maisonette' in text:
        result['is_multi_floor'] = True
        result['num_floors_span'] = max(result['num_floors_span'], 2)
    if 'duplex' in text:
        result['is_multi_floor'] = True
        result['num_floors_span'] = max(result['num_floors_span'], 2)
    if 'triplex' in text:
        result['is_multi_floor'] = True
        result['num_floors_span'] = max(result['num_floors_span'], 3)

    # Assign floor category based on level
    if result['floor_category'] is None and result['floor_level'] is not None:
        level = result['floor_level']
        if level < 0:
            result['floor_category'] = 'basement'
        elif level == 0:
            result['floor_category'] = 'ground'
        elif level <= 2:
            result['floor_category'] = 'lower'
        elif level <= 5:
            result['floor_category'] = 'mid'
        else:
            result['floor_category'] = 'high'

    return result


def load_and_enrich_data():
    """Load listings and extract floor information."""
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT
            id, source, property_id, address, postcode,
            price_pcm, bedrooms, size_sqft, property_type,
            summary, description, features,
            has_first_floor, has_second_floor, has_third_floor, floor_count
        FROM listings
        WHERE price_pcm > 0
    """, conn)
    conn.close()

    print(f"Loaded {len(df)} listings")

    # Extract floor info for each row
    floor_data = df.apply(extract_floor_info, axis=1)
    floor_df = pd.DataFrame(floor_data.tolist())

    # Combine
    df = pd.concat([df, floor_df], axis=1)

    # Calculate price per sqft
    df['ppsf'] = df.apply(
        lambda r: r['price_pcm'] / r['size_sqft'] if r['size_sqft'] and r['size_sqft'] > 0 else None,
        axis=1
    )

    return df


def analyze_floor_impact(df: pd.DataFrame):
    """Analyze price impact by floor characteristics."""

    print("\n" + "=" * 70)
    print("FLOOR LEVEL ANALYSIS")
    print("=" * 70)

    # Filter to properties with sqft data
    df_sqft = df[df['ppsf'].notna() & (df['ppsf'] < 20)].copy()  # Exclude outliers

    # 1. Coverage Analysis
    print("\n[1] DATA COVERAGE")
    print("-" * 50)

    total = len(df_sqft)
    has_floor = len(df_sqft[df_sqft['floor_category'].notna()])
    has_level = len(df_sqft[df_sqft['floor_level'].notna()])
    has_lift = len(df_sqft[df_sqft['has_lift'].notna()])
    is_multi = len(df_sqft[df_sqft['is_multi_floor'] == True])

    print(f"Total with sqft data:     {total:,}")
    print(f"Has floor category:       {has_floor:,} ({100*has_floor/total:.1f}%)")
    print(f"Has numeric floor level:  {has_level:,} ({100*has_level/total:.1f}%)")
    print(f"Has lift information:     {has_lift:,} ({100*has_lift/total:.1f}%)")
    print(f"Multi-floor properties:   {is_multi:,} ({100*is_multi/total:.1f}%)")

    # 2. Floor Category Analysis
    print("\n[2] PRICE BY FLOOR CATEGORY")
    print("-" * 50)

    baseline_ppsf = df_sqft['ppsf'].median()
    print(f"Baseline median £/sqft: £{baseline_ppsf:.2f}")
    print()

    categories = ['penthouse', 'high', 'top_floor', 'mid', 'lower', 'ground', 'lower_ground', 'basement']

    print(f"{'Category':<15} {'Count':>8} {'Med Price':>12} {'Med Sqft':>10} {'Med £/sqft':>12} {'vs Baseline':>12}")
    print("-" * 70)

    for cat in categories:
        data = df_sqft[df_sqft['floor_category'] == cat]
        if len(data) >= 5:
            med_price = data['price_pcm'].median()
            med_sqft = data['size_sqft'].median()
            med_ppsf = data['ppsf'].median()
            vs_base = 100 * (med_ppsf / baseline_ppsf - 1)
            print(f"{cat:<15} {len(data):>8} £{med_price:>10,.0f} {med_sqft:>10,.0f} £{med_ppsf:>11.2f} {vs_base:>+11.1f}%")

    # Unknown category
    unknown = df_sqft[df_sqft['floor_category'].isna()]
    if len(unknown) > 0:
        med_ppsf = unknown['ppsf'].median()
        vs_base = 100 * (med_ppsf / baseline_ppsf - 1)
        print(f"{'unknown':<15} {len(unknown):>8} £{unknown['price_pcm'].median():>10,.0f} {unknown['size_sqft'].median():>10,.0f} £{med_ppsf:>11.2f} {vs_base:>+11.1f}%")

    # 3. Numeric Floor Level Analysis
    print("\n[3] PRICE BY NUMERIC FLOOR LEVEL")
    print("-" * 50)

    df_level = df_sqft[df_sqft['floor_level'].notna()].copy()
    df_level['floor_int'] = df_level['floor_level'].astype(int)

    print(f"{'Floor':>8} {'Count':>8} {'Med £/sqft':>12} {'vs Baseline':>12}")
    print("-" * 45)

    for floor in sorted(df_level['floor_int'].unique()):
        data = df_level[df_level['floor_int'] == floor]
        if len(data) >= 3:
            med_ppsf = data['ppsf'].median()
            vs_base = 100 * (med_ppsf / baseline_ppsf - 1)
            floor_name = {-1: 'Basement', 0: 'Ground'}.get(floor, f'Floor {floor}')
            print(f"{floor_name:>8} {len(data):>8} £{med_ppsf:>11.2f} {vs_base:>+11.1f}%")

    # 4. Lift Impact Analysis
    print("\n[4] LIFT ACCESS IMPACT")
    print("-" * 50)

    df_lift = df_sqft[df_sqft['has_lift'].notna()].copy()

    print(f"{'Lift':>10} {'Count':>8} {'Med £/sqft':>12} {'Avg Floor':>12}")
    print("-" * 45)

    for has_lift in [True, False]:
        data = df_lift[df_lift['has_lift'] == has_lift]
        if len(data) > 0:
            med_ppsf = data['ppsf'].median()
            avg_floor = data[data['floor_level'].notna()]['floor_level'].mean()
            lift_str = "With Lift" if has_lift else "No Lift"
            print(f"{lift_str:>10} {len(data):>8} £{med_ppsf:>11.2f} {avg_floor:>11.1f}")

    # Lift impact by floor level
    print("\n  Lift impact by floor level:")
    df_lift_level = df_lift[df_lift['floor_level'].notna()].copy()

    for floor_range, label in [((0, 2), 'Floors 0-2'), ((3, 5), 'Floors 3-5'), ((6, 20), 'Floors 6+')]:
        subset = df_lift_level[(df_lift_level['floor_level'] >= floor_range[0]) &
                               (df_lift_level['floor_level'] <= floor_range[1])]
        if len(subset) >= 10:
            with_lift = subset[subset['has_lift'] == True]['ppsf'].median()
            no_lift = subset[subset['has_lift'] == False]['ppsf'].median()
            if pd.notna(with_lift) and pd.notna(no_lift):
                diff = 100 * (with_lift / no_lift - 1)
                print(f"    {label}: Lift premium = {diff:+.1f}%")

    # 5. Multi-Floor Properties
    print("\n[5] MULTI-FLOOR PROPERTIES (Maisonettes/Duplexes)")
    print("-" * 50)

    single = df_sqft[df_sqft['is_multi_floor'] == False]
    multi = df_sqft[df_sqft['is_multi_floor'] == True]

    print(f"{'Type':<20} {'Count':>8} {'Med Price':>12} {'Med Sqft':>10} {'Med £/sqft':>12}")
    print("-" * 65)

    for name, data in [('Single floor', single), ('Multi-floor', multi)]:
        if len(data) > 0:
            print(f"{name:<20} {len(data):>8} £{data['price_pcm'].median():>10,.0f} {data['size_sqft'].median():>10,.0f} £{data['ppsf'].median():>11.2f}")

    # Multi-floor by number of floors
    print("\n  By number of floors spanned:")
    for floors in [2, 3, 4, 5]:
        data = df_sqft[df_sqft['num_floors_span'] == floors]
        if len(data) >= 5:
            print(f"    {floors} floors: n={len(data)}, median £/sqft = £{data['ppsf'].median():.2f}")

    # 6. Special Floor Types
    print("\n[6] SPECIAL FLOOR TYPE ANALYSIS")
    print("-" * 50)

    special_types = [
        ('is_penthouse', 'Penthouse'),
        ('is_basement', 'Basement'),
        ('is_lower_ground', 'Lower Ground'),
        ('is_ground', 'Ground Floor'),
        ('is_top_floor', 'Top Floor')
    ]

    for col, name in special_types:
        data = df_sqft[df_sqft[col] == True]
        if len(data) >= 3:
            vs_base = 100 * (data['ppsf'].median() / baseline_ppsf - 1)
            print(f"{name:<15}: n={len(data):>4}, median £/sqft = £{data['ppsf'].median():.2f} ({vs_base:+.1f}% vs baseline)")

    return df_sqft


def regression_analysis(df: pd.DataFrame):
    """Run regression to quantify floor premium controlling for other factors."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder

    print("\n" + "=" * 70)
    print("REGRESSION ANALYSIS: Floor Premium Controlling for Size/Location")
    print("=" * 70)

    # Filter to records with good data
    df_reg = df[
        (df['ppsf'].notna()) &
        (df['size_sqft'] > 0) &
        (df['bedrooms'] > 0) &
        (df['postcode'].notna())
    ].copy()

    # Create features
    df_reg['log_sqft'] = np.log1p(df_reg['size_sqft'])
    df_reg['log_price'] = np.log1p(df_reg['price_pcm'])

    # Encode postcode district
    df_reg['postcode_district'] = df_reg['postcode'].str.extract(r'^([A-Z]+\d+)')[0]
    le = LabelEncoder()
    df_reg['postcode_encoded'] = le.fit_transform(df_reg['postcode_district'].fillna('Unknown'))

    # Floor dummies
    df_reg['is_penthouse_num'] = df_reg['is_penthouse'].astype(int)
    df_reg['is_basement_num'] = df_reg['is_basement'].astype(int)
    df_reg['is_lower_ground_num'] = df_reg['is_lower_ground'].astype(int)
    df_reg['is_ground_num'] = df_reg['is_ground'].astype(int)
    df_reg['is_high_floor'] = (df_reg['floor_level'].fillna(0) >= 6).astype(int)
    df_reg['is_multi_floor_num'] = df_reg['is_multi_floor'].astype(int)
    df_reg['has_lift_num'] = df_reg['has_lift'].fillna(False).astype(int)

    # Features for regression
    feature_cols = [
        'log_sqft', 'bedrooms', 'postcode_encoded',
        'is_penthouse_num', 'is_basement_num', 'is_lower_ground_num',
        'is_ground_num', 'is_high_floor', 'is_multi_floor_num', 'has_lift_num'
    ]

    X = df_reg[feature_cols].fillna(0)
    y = df_reg['log_price']

    model = LinearRegression()
    model.fit(X, y)

    print(f"\nModel R² (log price): {model.score(X, y):.3f}")
    print(f"Sample size: {len(X)}")
    print("\nFloor Premium Coefficients (in log-price space):")
    print("-" * 50)

    floor_features = [
        ('is_penthouse_num', 'Penthouse'),
        ('is_high_floor', 'High Floor (6+)'),
        ('has_lift_num', 'Has Lift'),
        ('is_multi_floor_num', 'Multi-Floor'),
        ('is_ground_num', 'Ground Floor'),
        ('is_lower_ground_num', 'Lower Ground'),
        ('is_basement_num', 'Basement'),
    ]

    for feat, name in floor_features:
        idx = feature_cols.index(feat)
        coef = model.coef_[idx]
        pct_impact = 100 * (np.exp(coef) - 1)
        print(f"  {name:<20}: {pct_impact:+6.1f}% price impact")

    return model


def generate_feature_recommendations():
    """Generate recommendations for model features."""

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR PRICE PREDICTION MODEL")
    print("=" * 70)

    print("""
[RECOMMENDED FEATURES TO ADD]

1. floor_level_numeric (continuous)
   - Extract from text: 0=ground, 1=first, etc.
   - Use -0.5 for lower ground, -1 for basement
   - Penthouse: use building's top floor if known, else 10

2. floor_category (categorical → one-hot)
   - basement, lower_ground, ground, lower (1-2), mid (3-5), high (6+), penthouse

3. is_penthouse (binary)
   - Strong premium signal (~25-40%)

4. is_basement_or_lower_ground (binary)
   - Discount signal (~10-20%)

5. has_lift (binary)
   - Premium, especially for higher floors

6. lift_floor_interaction
   - high_floor * has_lift (captures that lift matters more at height)

7. is_multi_floor (binary)
   - Maisonettes/duplexes may have different pricing

8. num_floors_span (1, 2, 3, 4...)
   - More floors = more space but also more stairs

[FEATURE ENGINEERING CODE]

```python
def extract_floor_features(row):
    '''Add to V8 model feature engineering'''
    features = {}

    # Extract from text
    floor_info = extract_floor_info(row)

    # Numeric floor (impute 2 for unknown - typical apartment)
    features['floor_level'] = floor_info['floor_level'] or 2

    # Categories
    features['is_penthouse'] = int(floor_info['is_penthouse'])
    features['is_basement'] = int(floor_info['is_basement'] or floor_info['is_lower_ground'])
    features['is_ground'] = int(floor_info['is_ground'])
    features['is_high_floor'] = int((floor_info['floor_level'] or 0) >= 6)

    # Lift interaction
    features['has_lift'] = int(floor_info['has_lift'] or False)
    features['high_floor_no_lift'] = features['is_high_floor'] * (1 - features['has_lift'])

    # Multi-floor
    features['is_multi_floor'] = int(floor_info['is_multi_floor'])
    features['floors_span'] = floor_info['num_floors_span']

    return features
```

[EXPECTED IMPACT]

Based on analysis:
- Penthouse premium: +25-40% vs baseline
- High floor (6+) premium: +5-15%
- Ground floor discount: -5-10%
- Basement/lower ground discount: -15-25%
- Lift premium on high floors: +10-20%

These features could improve model R² by 1-3% and reduce MAPE by 0.5-1%.
""")


def main():
    df = load_and_enrich_data()
    df_analyzed = analyze_floor_impact(df)
    regression_analysis(df_analyzed)
    generate_feature_recommendations()

    # Save enriched data for model use
    output_path = os.path.join(SCRIPT_DIR, 'output', 'floor_analysis_data.csv')
    df_analyzed.to_csv(output_path, index=False)
    print(f"\nSaved enriched data to: {output_path}")


if __name__ == '__main__':
    main()
