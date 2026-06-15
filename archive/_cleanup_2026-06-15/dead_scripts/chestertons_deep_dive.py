#!/usr/bin/env python3
"""
Deep Dive Cross-Sectional Analysis: Chestertons vs Market Average

Comprehensive comparison across:
- Price segments
- Areas/postcodes
- Property sizes
- Bedroom counts
- Days on market
- Price reduction patterns
- Price per sqft
- Listing freshness
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta
import re
import os
from scipy import stats

# Get absolute path to database
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'output', 'rentals.db')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def parse_rightmove_date(date_str: str, reference_date: datetime = None) -> datetime:
    """Parse Rightmove date strings."""
    if not date_str or pd.isna(date_str):
        return None
    if reference_date is None:
        reference_date = datetime.now()
    date_str = str(date_str).strip()
    if 'yesterday' in date_str.lower():
        return reference_date - timedelta(days=1)
    if 'today' in date_str.lower():
        return reference_date
    match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def normalize_agency(agent_name: str) -> str:
    """Normalize agent names to parent company."""
    if not agent_name or pd.isna(agent_name):
        return 'Other'
    agent = str(agent_name).lower().strip()

    agency_patterns = {
        'Chestertons': ['chestertons'],
        'Foxtons': ['foxtons'],
        'Knight Frank': ['knight frank'],
        'Savills': ['savills'],
        'Hamptons': ['hamptons'],
        'Marsh & Parsons': ['marsh & parsons'],
        'Dexters': ['dexters'],
        'John D Wood': ['john d wood'],
        'Winkworth': ['winkworth'],
        'KFH': ['kfh', 'draker kfh'],
        'JLL': ['jll'],
        'Strutt & Parker': ['strutt & parker'],
        'Sothebys': ['sotheby'],
        'Domus Nova': ['domus nova'],
        'OpenRent': ['openrent'],
        'Carter Jonas': ['carter jonas'],
    }

    for agency, patterns in agency_patterns.items():
        for pattern in patterns:
            if pattern in agent:
                return agency
    return 'Other'


def load_data():
    """Load all relevant data from database."""
    conn = sqlite3.connect(DB_PATH)

    # Load Rightmove data (has agent info + dates)
    rightmove = pd.read_sql("""
        SELECT
            source,
            property_id,
            address,
            postcode,
            area,
            price_pcm,
            bedrooms,
            bathrooms,
            size_sqft,
            property_type,
            furnished,
            added_date,
            scraped_at,
            agent_name,
            let_agreed
        FROM listings
        WHERE source = 'rightmove'
          AND price_pcm > 0
    """, conn)

    # Load direct Chestertons data
    chestertons_direct = pd.read_sql("""
        SELECT
            source,
            property_id,
            address,
            postcode,
            area,
            price_pcm,
            bedrooms,
            bathrooms,
            size_sqft,
            property_type,
            furnished,
            added_date,
            scraped_at,
            'Chestertons (Direct)' as agent_name,
            let_agreed
        FROM listings
        WHERE source = 'chestertons'
          AND price_pcm > 0
    """, conn)

    conn.close()

    # Combine
    df = pd.concat([rightmove, chestertons_direct], ignore_index=True)

    # Process dates
    df['scraped_datetime'] = pd.to_datetime(df['scraped_at'], errors='coerce')
    reference_date = df['scraped_datetime'].max()

    df['added_datetime'] = df['added_date'].apply(
        lambda x: parse_rightmove_date(x, reference_date)
    )

    df['days_on_market'] = df.apply(
        lambda row: max(0, (row['scraped_datetime'] - row['added_datetime']).days)
        if row['added_datetime'] and row['scraped_datetime'] else None,
        axis=1
    )

    # Normalize agency
    df['agency'] = df['agent_name'].apply(normalize_agency)

    # Mark Chestertons (both Rightmove and direct)
    df['is_chestertons'] = (df['agency'] == 'Chestertons') | (df['source'] == 'chestertons')

    # Calculate price per sqft
    df['ppsf'] = df.apply(
        lambda r: r['price_pcm'] / r['size_sqft'] if r['size_sqft'] and r['size_sqft'] > 0 else None,
        axis=1
    )

    # Price buckets
    df['price_bucket'] = pd.cut(
        df['price_pcm'],
        bins=[0, 2000, 3000, 5000, 7500, 10000, 15000, 25000, float('inf')],
        labels=['<£2k', '£2-3k', '£3-5k', '£5-7.5k', '£7.5-10k', '£10-15k', '£15-25k', '£25k+']
    )

    # Size buckets
    df['size_bucket'] = pd.cut(
        df['size_sqft'],
        bins=[0, 500, 750, 1000, 1500, 2000, float('inf')],
        labels=['<500', '500-750', '750-1000', '1000-1500', '1500-2000', '2000+']
    )

    # Postcode district
    df['postcode_district'] = df['postcode'].str.extract(r'^([A-Z]+\d+)', expand=False)

    # Was price reduced
    df['was_reduced'] = df['added_date'].str.lower().str.contains('reduced', na=False)

    print(f"Loaded {len(df):,} total listings")
    print(f"  - Rightmove: {len(rightmove):,}")
    print(f"  - Chestertons Direct: {len(chestertons_direct):,}")
    print(f"  - Chestertons (all): {df['is_chestertons'].sum():,}")

    return df


def calculate_comparison_stats(df: pd.DataFrame, group_col: str, metric_col: str):
    """Calculate Chestertons vs Others for a given grouping."""
    chester = df[df['is_chestertons']]
    others = df[~df['is_chestertons']]

    chester_stats = chester.groupby(group_col)[metric_col].agg(['count', 'mean', 'median', 'std'])
    others_stats = others.groupby(group_col)[metric_col].agg(['count', 'mean', 'median', 'std'])

    chester_stats.columns = ['C_count', 'C_mean', 'C_median', 'C_std']
    others_stats.columns = ['O_count', 'O_mean', 'O_median', 'O_std']

    combined = chester_stats.join(others_stats, how='outer')
    combined['diff_pct'] = ((combined['C_median'] - combined['O_median']) / combined['O_median'] * 100).round(1)

    return combined


def section_header(title: str):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def analyze_overview(df: pd.DataFrame):
    """High-level overview comparison."""
    section_header("1. OVERVIEW: CHESTERTONS vs MARKET")

    chester = df[df['is_chestertons']]
    others = df[~df['is_chestertons']]

    metrics = {
        'Total Listings': (len(chester), len(others)),
        'With Sqft Data': (chester['size_sqft'].notna().sum(), others['size_sqft'].notna().sum()),
        'Sqft Coverage %': (100*chester['size_sqft'].notna().mean(), 100*others['size_sqft'].notna().mean()),
        'Median Price (£)': (chester['price_pcm'].median(), others['price_pcm'].median()),
        'Mean Price (£)': (chester['price_pcm'].mean(), others['price_pcm'].mean()),
        'Median Sqft': (chester['size_sqft'].median(), others['size_sqft'].median()),
        'Median £/sqft': (chester['ppsf'].median(), others['ppsf'].median()),
        'Median Bedrooms': (chester['bedrooms'].median(), others['bedrooms'].median()),
        'Price Reduction Rate %': (100*chester['was_reduced'].mean(), 100*others['was_reduced'].mean()),
    }

    # Days on market (Rightmove only has dates)
    chester_rm = chester[chester['days_on_market'].notna()]
    others_rm = others[others['days_on_market'].notna()]
    if len(chester_rm) > 0:
        metrics['Median Days on Market'] = (chester_rm['days_on_market'].median(), others_rm['days_on_market'].median())

    print(f"\n{'Metric':<30} {'Chestertons':>15} {'Others':>15} {'Diff %':>12}")
    print("-" * 75)

    for metric, (c_val, o_val) in metrics.items():
        if o_val and o_val != 0:
            diff = (c_val - o_val) / o_val * 100
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "N/A"

        if 'Price' in metric or '£' in metric:
            print(f"{metric:<30} {c_val:>15,.0f} {o_val:>15,.0f} {diff_str:>12}")
        elif '%' in metric:
            print(f"{metric:<30} {c_val:>15.1f}% {o_val:>15.1f}% {diff_str:>12}")
        else:
            print(f"{metric:<30} {c_val:>15,.1f} {o_val:>15,.1f} {diff_str:>12}")


def analyze_by_price_segment(df: pd.DataFrame):
    """Compare performance across price segments."""
    section_header("2. PERFORMANCE BY PRICE SEGMENT")

    chester = df[df['is_chestertons']]
    others = df[~df['is_chestertons']]

    bucket_order = ['<£2k', '£2-3k', '£3-5k', '£5-7.5k', '£7.5-10k', '£10-15k', '£15-25k', '£25k+']

    print("\n2a. LISTING DISTRIBUTION BY PRICE")
    print(f"{'Price Bucket':<15} {'Chester #':>12} {'Chester %':>12} {'Others #':>12} {'Others %':>12} {'Δ Share':>10}")
    print("-" * 75)

    c_counts = chester['price_bucket'].value_counts()
    o_counts = others['price_bucket'].value_counts()

    for bucket in bucket_order:
        c_n = c_counts.get(bucket, 0)
        o_n = o_counts.get(bucket, 0)
        c_pct = 100 * c_n / len(chester) if len(chester) > 0 else 0
        o_pct = 100 * o_n / len(others) if len(others) > 0 else 0
        delta = c_pct - o_pct
        print(f"{bucket:<15} {c_n:>12,} {c_pct:>11.1f}% {o_n:>12,} {o_pct:>11.1f}% {delta:>+9.1f}pp")

    # Days on market by price segment
    print("\n2b. MEDIAN DAYS ON MARKET BY PRICE (Rightmove data only)")
    print(f"{'Price Bucket':<15} {'Chester':>12} {'Others':>12} {'Difference':>15}")
    print("-" * 55)

    chester_rm = chester[chester['days_on_market'].notna()]
    others_rm = others[others['days_on_market'].notna()]

    for bucket in bucket_order:
        c_days = chester_rm[chester_rm['price_bucket'] == bucket]['days_on_market'].median()
        o_days = others_rm[others_rm['price_bucket'] == bucket]['days_on_market'].median()

        if pd.notna(c_days) and pd.notna(o_days):
            diff = c_days - o_days
            diff_str = f"{diff:+.0f} days"
        else:
            diff_str = "N/A"

        c_str = f"{c_days:.0f}d" if pd.notna(c_days) else "N/A"
        o_str = f"{o_days:.0f}d" if pd.notna(o_days) else "N/A"
        print(f"{bucket:<15} {c_str:>12} {o_str:>12} {diff_str:>15}")


def analyze_by_area(df: pd.DataFrame):
    """Compare performance across areas/postcodes."""
    section_header("3. PERFORMANCE BY AREA")

    chester = df[df['is_chestertons']]
    others = df[~df['is_chestertons']]

    # Top postcodes where Chestertons is active
    chester_postcodes = chester['postcode_district'].value_counts().head(15)

    print("\n3a. CHESTERTONS TOP AREAS (by listing count)")
    print(f"{'Postcode':<10} {'Chester #':>10} {'Others #':>10} {'C Med Price':>12} {'O Med Price':>12} {'Price Δ':>10}")
    print("-" * 70)

    for postcode in chester_postcodes.index:
        c_data = chester[chester['postcode_district'] == postcode]
        o_data = others[others['postcode_district'] == postcode]

        c_n = len(c_data)
        o_n = len(o_data)
        c_price = c_data['price_pcm'].median()
        o_price = o_data['price_pcm'].median() if len(o_data) > 0 else None

        if o_price and o_price > 0:
            diff = (c_price - o_price) / o_price * 100
            diff_str = f"{diff:+.1f}%"
            o_price_str = f"£{o_price:,.0f}"
        else:
            diff_str = "N/A"
            o_price_str = "N/A"

        print(f"{postcode:<10} {c_n:>10,} {o_n:>10,} £{c_price:>10,.0f} {o_price_str:>12} {diff_str:>10}")

    # Market share by area
    print("\n3b. CHESTERTONS MARKET SHARE BY AREA")
    print(f"{'Postcode':<10} {'Total':>10} {'Chester':>10} {'Share %':>10}")
    print("-" * 45)

    all_postcodes = df['postcode_district'].value_counts().head(15)
    for postcode in all_postcodes.index:
        total = len(df[df['postcode_district'] == postcode])
        c_count = len(chester[chester['postcode_district'] == postcode])
        share = 100 * c_count / total if total > 0 else 0
        print(f"{postcode:<10} {total:>10,} {c_count:>10,} {share:>9.1f}%")


def analyze_by_size(df: pd.DataFrame):
    """Compare performance across property sizes."""
    section_header("4. PERFORMANCE BY PROPERTY SIZE")

    # Filter to records with sqft
    df_sqft = df[df['size_sqft'].notna() & (df['size_sqft'] > 0)]
    chester = df_sqft[df_sqft['is_chestertons']]
    others = df_sqft[~df_sqft['is_chestertons']]

    bucket_order = ['<500', '500-750', '750-1000', '1000-1500', '1500-2000', '2000+']

    print("\n4a. LISTING DISTRIBUTION BY SIZE")
    print(f"{'Size (sqft)':<15} {'Chester #':>10} {'Chester %':>10} {'Others #':>10} {'Others %':>10}")
    print("-" * 60)

    c_counts = chester['size_bucket'].value_counts()
    o_counts = others['size_bucket'].value_counts()

    for bucket in bucket_order:
        c_n = c_counts.get(bucket, 0)
        o_n = o_counts.get(bucket, 0)
        c_pct = 100 * c_n / len(chester) if len(chester) > 0 else 0
        o_pct = 100 * o_n / len(others) if len(others) > 0 else 0
        print(f"{bucket:<15} {c_n:>10,} {c_pct:>9.1f}% {o_n:>10,} {o_pct:>9.1f}%")

    print("\n4b. PRICE PER SQFT BY SIZE SEGMENT")
    print(f"{'Size (sqft)':<15} {'C £/sqft':>12} {'O £/sqft':>12} {'Difference':>12}")
    print("-" * 55)

    for bucket in bucket_order:
        c_ppsf = chester[chester['size_bucket'] == bucket]['ppsf'].median()
        o_ppsf = others[others['size_bucket'] == bucket]['ppsf'].median()

        if pd.notna(c_ppsf) and pd.notna(o_ppsf) and o_ppsf > 0:
            diff = (c_ppsf - o_ppsf) / o_ppsf * 100
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "N/A"

        c_str = f"£{c_ppsf:.2f}" if pd.notna(c_ppsf) else "N/A"
        o_str = f"£{o_ppsf:.2f}" if pd.notna(o_ppsf) else "N/A"
        print(f"{bucket:<15} {c_str:>12} {o_str:>12} {diff_str:>12}")


def analyze_by_bedrooms(df: pd.DataFrame):
    """Compare performance across bedroom counts."""
    section_header("5. PERFORMANCE BY BEDROOM COUNT")

    df_beds = df[df['bedrooms'].notna()]
    chester = df_beds[df_beds['is_chestertons']]
    others = df_beds[~df_beds['is_chestertons']]

    print("\n5a. DISTRIBUTION AND PRICING BY BEDROOMS")
    print(f"{'Beds':<8} {'C #':>8} {'C %':>8} {'O #':>8} {'O %':>8} {'C Price':>12} {'O Price':>12} {'Δ':>8}")
    print("-" * 80)

    for beds in sorted(df_beds['bedrooms'].dropna().unique()):
        if beds > 6:
            continue
        c_data = chester[chester['bedrooms'] == beds]
        o_data = others[others['bedrooms'] == beds]

        c_n = len(c_data)
        o_n = len(o_data)
        c_pct = 100 * c_n / len(chester) if len(chester) > 0 else 0
        o_pct = 100 * o_n / len(others) if len(others) > 0 else 0
        c_price = c_data['price_pcm'].median()
        o_price = o_data['price_pcm'].median()

        if o_price > 0:
            diff = (c_price - o_price) / o_price * 100
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "N/A"

        print(f"{int(beds):<8} {c_n:>8,} {c_pct:>7.1f}% {o_n:>8,} {o_pct:>7.1f}% £{c_price:>10,.0f} £{o_price:>10,.0f} {diff_str:>8}")


def analyze_pricing_strategy(df: pd.DataFrame):
    """Analyze pricing strategy and reduction patterns."""
    section_header("6. PRICING STRATEGY ANALYSIS")

    chester = df[df['is_chestertons']]
    others = df[~df['is_chestertons']]

    # Price reduction analysis
    print("\n6a. PRICE REDUCTION PATTERNS")

    c_reduced = chester['was_reduced'].sum()
    c_total = len(chester[chester['was_reduced'].notna()])
    o_reduced = others['was_reduced'].sum()
    o_total = len(others[others['was_reduced'].notna()])

    print(f"Chestertons: {c_reduced:,} / {c_total:,} = {100*c_reduced/c_total:.1f}% had price reductions")
    print(f"Others:      {o_reduced:,} / {o_total:,} = {100*o_reduced/o_total:.1f}% had price reductions")

    # Reduction rate by price bucket
    print("\n6b. REDUCTION RATE BY PRICE SEGMENT")
    print(f"{'Price Bucket':<15} {'Chester %':>12} {'Others %':>12} {'Difference':>12}")
    print("-" * 55)

    bucket_order = ['<£2k', '£2-3k', '£3-5k', '£5-7.5k', '£7.5-10k', '£10-15k', '£15-25k', '£25k+']

    for bucket in bucket_order:
        c_data = chester[chester['price_bucket'] == bucket]
        o_data = others[others['price_bucket'] == bucket]

        c_rate = 100 * c_data['was_reduced'].mean() if len(c_data) > 0 else 0
        o_rate = 100 * o_data['was_reduced'].mean() if len(o_data) > 0 else 0

        diff = c_rate - o_rate
        print(f"{bucket:<15} {c_rate:>11.1f}% {o_rate:>11.1f}% {diff:>+11.1f}pp")

    # Price positioning analysis
    print("\n6c. PRICE POSITIONING (vs market median by area)")

    # Calculate area medians
    area_medians = df.groupby('postcode_district')['price_pcm'].median()
    df_with_median = df.copy()
    df_with_median['area_median'] = df_with_median['postcode_district'].map(area_medians)
    df_with_median['price_vs_median'] = (df_with_median['price_pcm'] / df_with_median['area_median'] - 1) * 100

    chester_pos = df_with_median[df_with_median['is_chestertons']]['price_vs_median']
    others_pos = df_with_median[~df_with_median['is_chestertons']]['price_vs_median']

    print(f"\nChestertons avg price position: {chester_pos.mean():+.1f}% vs area median")
    print(f"Others avg price position:      {others_pos.mean():+.1f}% vs area median")

    # Distribution of price positioning
    print("\nPrice positioning distribution:")
    print(f"{'Position':<25} {'Chester %':>12} {'Others %':>12}")
    print("-" * 50)

    for label, (low, high) in [
        ('> 20% below median', (-float('inf'), -20)),
        ('10-20% below median', (-20, -10)),
        ('Within ±10% of median', (-10, 10)),
        ('10-20% above median', (10, 20)),
        ('> 20% above median', (20, float('inf'))),
    ]:
        c_pct = 100 * ((chester_pos >= low) & (chester_pos < high)).mean()
        o_pct = 100 * ((others_pos >= low) & (others_pos < high)).mean()
        print(f"{label:<25} {c_pct:>11.1f}% {o_pct:>11.1f}%")


def analyze_time_on_market(df: pd.DataFrame):
    """Detailed time-on-market analysis."""
    section_header("7. TIME ON MARKET DEEP DIVE (Rightmove only)")

    # Filter to Rightmove only (only source with dates) and exclude 0-day listings
    df_rm = df[(df['source'] == 'rightmove') & (df['days_on_market'].notna()) & (df['days_on_market'] > 0)].copy()
    chester = df_rm[df_rm['agency'] == 'Chestertons']
    others = df_rm[df_rm['agency'] != 'Chestertons']

    print(f"\nData available: {len(chester):,} Chestertons, {len(others):,} Others (Rightmove listings with dates)")

    # Percentile comparison
    print("\n7a. DAYS ON MARKET DISTRIBUTION")
    print(f"{'Percentile':<15} {'Chester':>12} {'Others':>12} {'Difference':>12}")
    print("-" * 55)

    for pct in [10, 25, 50, 75, 90, 95]:
        c_val = np.percentile(chester['days_on_market'], pct)
        o_val = np.percentile(others['days_on_market'], pct)
        diff = c_val - o_val
        print(f"{pct}th percentile {c_val:>11.0f}d {o_val:>11.0f}d {diff:>+11.0f}d")

    # Statistical test
    stat, pval = stats.mannwhitneyu(chester['days_on_market'], others['days_on_market'], alternative='two-sided')
    print(f"\nMann-Whitney U test: p-value = {pval:.4f}")
    if pval < 0.05:
        print("→ Statistically significant difference in days on market")
    else:
        print("→ No statistically significant difference")

    # Stale listing analysis
    print("\n7b. STALE LISTING RATES")
    print(f"{'Threshold':<20} {'Chester %':>12} {'Others %':>12} {'Difference':>12}")
    print("-" * 60)

    for threshold in [14, 21, 30, 45, 60]:
        c_rate = 100 * (chester['days_on_market'] >= threshold).mean()
        o_rate = 100 * (others['days_on_market'] >= threshold).mean()
        diff = c_rate - o_rate
        print(f">{threshold} days {c_rate:>17.1f}% {o_rate:>11.1f}% {diff:>+11.1f}pp")


def analyze_competitive_position(df: pd.DataFrame):
    """Compare Chestertons against specific competitors."""
    section_header("8. COMPETITIVE POSITIONING (Rightmove only)")

    competitors = ['Foxtons', 'Knight Frank', 'Savills', 'Hamptons', 'Marsh & Parsons', 'Dexters']

    # Use Rightmove only for fair comparison (only source with dates)
    df_rm = df[(df['source'] == 'rightmove') & (df['days_on_market'].notna()) & (df['days_on_market'] > 0)].copy()
    chester = df_rm[df_rm['agency'] == 'Chestertons']

    print("\n8a. CHESTERTONS vs KEY COMPETITORS")
    print(f"{'Agency':<20} {'Count':>8} {'Med Price':>12} {'Med Days':>10} {'Red Rate':>10}")
    print("-" * 65)

    # Chestertons first
    c_count = len(chester)
    c_price = chester['price_pcm'].median()
    c_days = chester['days_on_market'].median()
    c_red = 100 * chester['was_reduced'].mean()
    print(f"{'CHESTERTONS':<20} {c_count:>8,} £{c_price:>10,.0f} {c_days:>9.0f}d {c_red:>9.1f}%")
    print("-" * 65)

    for comp in competitors:
        comp_data = df_rm[df_rm['agency'] == comp]
        if len(comp_data) > 10:
            print(f"{comp:<20} {len(comp_data):>8,} £{comp_data['price_pcm'].median():>10,.0f} {comp_data['days_on_market'].median():>9.0f}d {100*comp_data['was_reduced'].mean():>9.1f}%")

    # Value proposition analysis
    print("\n8b. VALUE ANALYSIS: PRICE PER SQFT COMPARISON")

    df_sqft = df[(df['size_sqft'].notna()) & (df['size_sqft'] > 0)]

    print(f"{'Agency':<20} {'Med £/sqft':>12} {'vs Chester':>12}")
    print("-" * 50)

    chester_ppsf = df_sqft[df_sqft['is_chestertons']]['ppsf'].median()
    print(f"{'CHESTERTONS':<20} £{chester_ppsf:>10.2f} {'baseline':>12}")

    for comp in competitors:
        comp_data = df_sqft[df_sqft['agency'] == comp]
        if len(comp_data) > 10:
            comp_ppsf = comp_data['ppsf'].median()
            diff = (comp_ppsf - chester_ppsf) / chester_ppsf * 100
            print(f"{comp:<20} £{comp_ppsf:>10.2f} {diff:>+11.1f}%")


def create_visualizations(df: pd.DataFrame):
    """Create comprehensive visualization dashboard."""

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('Chestertons Deep Dive: Cross-Sectional Analysis vs Market',
                 fontsize=18, fontweight='bold', y=0.995)

    # Setup grid
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    chester = df[df['is_chestertons']]
    others = df[~df['is_chestertons']]

    # For time-on-market: use ONLY Rightmove data (has dates)
    # Filter to records with actual positive days (exclude 0 which means no date)
    df_rm = df[df['source'] == 'rightmove'].copy()
    chester_rm = df_rm[df_rm['agency'] == 'Chestertons']
    others_rm = df_rm[df_rm['agency'] != 'Chestertons']

    # Color scheme
    chester_color = '#2E86AB'  # Blue
    others_color = '#A23B72'   # Purple

    # 1. Price Distribution Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 20000, 40)
    ax1.hist(chester['price_pcm'], bins=bins, alpha=0.6, label=f'Chestertons (n={len(chester):,})',
             color=chester_color, density=True)
    ax1.hist(others['price_pcm'], bins=bins, alpha=0.6, label=f'Others (n={len(others):,})',
             color=others_color, density=True)
    ax1.axvline(chester['price_pcm'].median(), color=chester_color, linestyle='--', linewidth=2)
    ax1.axvline(others['price_pcm'].median(), color=others_color, linestyle='--', linewidth=2)
    ax1.set_xlabel('Monthly Rent (£)')
    ax1.set_ylabel('Density')
    ax1.set_title('Price Distribution')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 20000)

    # 2. Market Share by Price Segment
    ax2 = fig.add_subplot(gs[0, 1])
    bucket_order = ['<£2k', '£2-3k', '£3-5k', '£5-7.5k', '£7.5-10k', '£10-15k', '£15-25k', '£25k+']

    shares = []
    for bucket in bucket_order:
        total = len(df[df['price_bucket'] == bucket])
        c_count = len(chester[chester['price_bucket'] == bucket])
        shares.append(100 * c_count / total if total > 0 else 0)

    bars = ax2.bar(range(len(bucket_order)), shares, color=chester_color, edgecolor='white')
    ax2.axhline(100 * len(chester) / len(df), color='red', linestyle='--',
                label=f'Overall Share: {100*len(chester)/len(df):.1f}%')
    ax2.set_xticks(range(len(bucket_order)))
    ax2.set_xticklabels(bucket_order, rotation=45, ha='right')
    ax2.set_ylabel('Chestertons Market Share (%)')
    ax2.set_title('Market Share by Price Segment')
    ax2.legend()
    for bar, val in zip(bars, shares):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # 3. Days on Market Comparison (Rightmove only - has dates)
    ax3 = fig.add_subplot(gs[0, 2])
    chester_dom = chester_rm[chester_rm['days_on_market'].notna() & (chester_rm['days_on_market'] > 0)]['days_on_market']
    others_dom = others_rm[others_rm['days_on_market'].notna() & (others_rm['days_on_market'] > 0)]['days_on_market']

    bp = ax3.boxplot([chester_dom, others_dom], tick_labels=['Chestertons\n(n={})'.format(len(chester_dom)), 'Others\n(n={})'.format(len(others_dom))],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor(chester_color)
    bp['boxes'][1].set_facecolor(others_color)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax3.set_ylabel('Days on Market')
    ax3.set_title('Time to Let (Rightmove data)')
    ax3.set_ylim(0, 60)

    # Add medians as text
    ax3.text(1, chester_dom.median() + 2, f'{chester_dom.median():.0f}d', ha='center', fontsize=10, fontweight='bold')
    ax3.text(2, others_dom.median() + 2, f'{others_dom.median():.0f}d', ha='center', fontsize=10, fontweight='bold')

    # 4. Days on Market by Price Bucket (Rightmove only)
    ax4 = fig.add_subplot(gs[1, 0])

    c_days = []
    o_days = []
    for bucket in bucket_order:
        c_data = chester_rm[(chester_rm['price_bucket'] == bucket) & (chester_rm['days_on_market'].notna()) & (chester_rm['days_on_market'] > 0)]
        o_data = others_rm[(others_rm['price_bucket'] == bucket) & (others_rm['days_on_market'].notna()) & (others_rm['days_on_market'] > 0)]
        c_days.append(c_data['days_on_market'].median() if len(c_data) > 3 else np.nan)
        o_days.append(o_data['days_on_market'].median() if len(o_data) > 5 else np.nan)

    x = np.arange(len(bucket_order))
    width = 0.35
    ax4.bar(x - width/2, c_days, width, label='Chestertons', color=chester_color, alpha=0.8)
    ax4.bar(x + width/2, o_days, width, label='Others', color=others_color, alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(bucket_order, rotation=45, ha='right')
    ax4.set_ylabel('Median Days on Market')
    ax4.set_title('Days on Market by Price (Rightmove)')
    ax4.legend()

    # 5. Price per Sqft Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    chester_ppsf = chester[chester['ppsf'].notna() & (chester['ppsf'] < 20)]['ppsf']
    others_ppsf = others[others['ppsf'].notna() & (others['ppsf'] < 20)]['ppsf']

    bins = np.linspace(0, 15, 30)
    ax5.hist(chester_ppsf, bins=bins, alpha=0.6, label=f'Chestertons', color=chester_color, density=True)
    ax5.hist(others_ppsf, bins=bins, alpha=0.6, label=f'Others', color=others_color, density=True)
    ax5.axvline(chester_ppsf.median(), color=chester_color, linestyle='--', linewidth=2)
    ax5.axvline(others_ppsf.median(), color=others_color, linestyle='--', linewidth=2)
    ax5.set_xlabel('Price per Sqft (£)')
    ax5.set_ylabel('Density')
    ax5.set_title(f'Price per Sqft Distribution\nChester: £{chester_ppsf.median():.2f} | Others: £{others_ppsf.median():.2f}')
    ax5.legend()

    # 6. Price Reduction Rate Comparison
    ax6 = fig.add_subplot(gs[1, 2])

    c_red = []
    o_red = []
    for bucket in bucket_order:
        c_data = chester[chester['price_bucket'] == bucket]
        o_data = others[others['price_bucket'] == bucket]
        c_red.append(100 * c_data['was_reduced'].mean() if len(c_data) > 5 else np.nan)
        o_red.append(100 * o_data['was_reduced'].mean() if len(o_data) > 5 else np.nan)

    x = np.arange(len(bucket_order))
    ax6.bar(x - width/2, c_red, width, label='Chestertons', color=chester_color, alpha=0.8)
    ax6.bar(x + width/2, o_red, width, label='Others', color=others_color, alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(bucket_order, rotation=45, ha='right')
    ax6.set_ylabel('Price Reduction Rate (%)')
    ax6.set_title('Price Reduction Rate by Segment')
    ax6.legend()

    # 7. Geographic Concentration (Market Share by Postcode)
    ax7 = fig.add_subplot(gs[2, 0])

    top_postcodes = df['postcode_district'].value_counts().head(12).index
    shares_by_area = []
    for pc in top_postcodes:
        total = len(df[df['postcode_district'] == pc])
        c_count = len(chester[chester['postcode_district'] == pc])
        shares_by_area.append(100 * c_count / total if total > 0 else 0)

    colors = [chester_color if s > 100*len(chester)/len(df) else '#888888' for s in shares_by_area]
    bars = ax7.barh(range(len(top_postcodes)), shares_by_area, color=colors, alpha=0.8)
    ax7.axvline(100 * len(chester) / len(df), color='red', linestyle='--', linewidth=2)
    ax7.set_yticks(range(len(top_postcodes)))
    ax7.set_yticklabels(top_postcodes)
    ax7.set_xlabel('Chestertons Market Share (%)')
    ax7.set_title('Market Share by Area\n(red line = overall avg)')

    # 8. Bedroom Mix Comparison
    ax8 = fig.add_subplot(gs[2, 1])

    beds_range = [1, 2, 3, 4, 5]
    c_pcts = []
    o_pcts = []
    for beds in beds_range:
        c_pcts.append(100 * (chester['bedrooms'] == beds).sum() / len(chester))
        o_pcts.append(100 * (others['bedrooms'] == beds).sum() / len(others))

    x = np.arange(len(beds_range))
    ax8.bar(x - width/2, c_pcts, width, label='Chestertons', color=chester_color, alpha=0.8)
    ax8.bar(x + width/2, o_pcts, width, label='Others', color=others_color, alpha=0.8)
    ax8.set_xticks(x)
    ax8.set_xticklabels([f'{b} bed' for b in beds_range])
    ax8.set_ylabel('% of Portfolio')
    ax8.set_title('Property Size Mix')
    ax8.legend()

    # 9. Competitor Comparison
    ax9 = fig.add_subplot(gs[2, 2])

    competitors = ['Chestertons', 'Foxtons', 'Knight Frank', 'Savills', 'Hamptons', 'Dexters']
    # Use Rightmove only for fair days comparison
    df_comp = df_rm[(df_rm['days_on_market'].notna()) & (df_rm['days_on_market'] > 0)]

    med_days = []
    med_prices = []
    counts = []
    for comp in competitors:
        data = df_comp[df_comp['agency'] == comp]
        med_days.append(data['days_on_market'].median() if len(data) > 10 else np.nan)
        med_prices.append(data['price_pcm'].median() if len(data) > 10 else np.nan)
        counts.append(len(data))

    colors_comp = [chester_color if c == 'Chestertons' else others_color for c in competitors]
    scatter = ax9.scatter(med_prices, med_days, s=[c*2 for c in counts], c=colors_comp, alpha=0.7, edgecolors='black')

    for i, comp in enumerate(competitors):
        if not np.isnan(med_prices[i]):
            ax9.annotate(comp, (med_prices[i], med_days[i]), fontsize=9,
                        xytext=(5, 5), textcoords='offset points')

    ax9.set_xlabel('Median Price (£/month)')
    ax9.set_ylabel('Median Days on Market')
    ax9.set_title('Competitive Position\n(bubble size = listing count)')
    ax9.grid(True, alpha=0.3)

    # 10. Summary Scorecard
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')

    # Calculate metrics for scorecard
    c_price = chester['price_pcm'].median()
    o_price = others['price_pcm'].median()
    # Use Rightmove data only for days on market (fair comparison)
    c_days_data = chester_rm[(chester_rm['days_on_market'].notna()) & (chester_rm['days_on_market'] > 0)]['days_on_market']
    o_days_data = others_rm[(others_rm['days_on_market'].notna()) & (others_rm['days_on_market'] > 0)]['days_on_market']
    c_days = c_days_data.median() if len(c_days_data) > 0 else 0
    o_days = o_days_data.median() if len(o_days_data) > 0 else 0
    c_ppsf = chester['ppsf'].median()
    o_ppsf = others['ppsf'].median()
    # Use Rightmove data for reduction rate (direct listings don't have this)
    c_red = 100 * chester_rm['was_reduced'].mean() if len(chester_rm) > 0 else 0
    o_red = 100 * others_rm['was_reduced'].mean() if len(others_rm) > 0 else 0
    c_sqft_cov = 100 * chester['size_sqft'].notna().mean()
    o_sqft_cov = 100 * others['size_sqft'].notna().mean()

    scorecard = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    CHESTERTONS PERFORMANCE SCORECARD                                      ║
╠═══════════════════════════════════════════╦══════════════════════════════════════════════════════════════╣
║  METRIC                                   ║  CHESTERTONS          MARKET AVG          VERDICT            ║
╠═══════════════════════════════════════════╬══════════════════════════════════════════════════════════════╣
║  Median Listing Price                     ║  £{c_price:>8,.0f}            £{o_price:>8,.0f}            {"⬆ Premium" if c_price > o_price else "⬇ Value":>18} ║
║  Median Days on Market                    ║  {c_days:>8.0f}d            {o_days:>8.0f}d            {"✓ Faster" if c_days < o_days else "✗ Slower":>18} ║
║  Median Price per Sqft                    ║  £{c_ppsf:>8.2f}            £{o_ppsf:>8.2f}            {"⬆ Premium" if c_ppsf > o_ppsf else "⬇ Value":>18} ║
║  Price Reduction Rate                     ║  {c_red:>8.1f}%            {o_red:>8.1f}%            {"✓ Lower" if c_red < o_red else "✗ Higher":>18} ║
║  Data Quality (Sqft Coverage)             ║  {c_sqft_cov:>8.1f}%            {o_sqft_cov:>8.1f}%            {"✓ Better" if c_sqft_cov > o_sqft_cov else "~ Similar":>18} ║
╚═══════════════════════════════════════════╩══════════════════════════════════════════════════════════════╝

KEY INSIGHTS:
• Chestertons focuses on the £{c_price/1000:.0f}k/month segment - {"premium" if c_price > o_price else "value"} positioning vs market
• Properties let in {c_days:.0f} days median vs market {o_days:.0f} days ({c_days - o_days:+.0f} day difference)
• Price reduction rate of {c_red:.1f}% indicates {"aggressive initial pricing" if c_red > o_red else "realistic initial pricing"}
• Strongest presence in: {', '.join(chester['postcode_district'].value_counts().head(5).index.tolist())}
"""

    ax10.text(0.5, 0.5, scorecard, transform=ax10.transAxes, fontsize=10,
              verticalalignment='center', horizontalalignment='center',
              fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.savefig(os.path.join(OUTPUT_DIR, 'chestertons_deep_dive.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nVisualization saved to: {os.path.join(OUTPUT_DIR, 'chestertons_deep_dive.png')}")


def main():
    """Run complete Chestertons deep dive analysis."""
    print("="*80)
    print(" CHESTERTONS DEEP DIVE ANALYSIS")
    print(" Cross-Sectional Comparison vs Market Average")
    print("="*80)

    df = load_data()

    analyze_overview(df)
    analyze_by_price_segment(df)
    analyze_by_area(df)
    analyze_by_size(df)
    analyze_by_bedrooms(df)
    analyze_pricing_strategy(df)
    analyze_time_on_market(df)
    analyze_competitive_position(df)

    create_visualizations(df)

    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
