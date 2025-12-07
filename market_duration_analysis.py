#!/usr/bin/env python3
"""
Market Duration Analysis - How long do properties stay on the market?

Analyzes days-on-market by:
- Agency/source
- Price buckets
- Property type
- Area/postcode

Note: Only Rightmove has added_date populated. Other sources lack this data.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import os

# Get absolute path to database
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'output', 'rentals.db')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def parse_rightmove_date(date_str: str, reference_date: datetime = None) -> datetime:
    """
    Parse Rightmove date strings like:
    - "Added yesterday"
    - "Added on 03/12/2025"
    - "Reduced on 27/11/2025"
    - "Reduced yesterday"

    Returns the actual date when property was added/reduced.
    """
    if not date_str or pd.isna(date_str):
        return None

    if reference_date is None:
        reference_date = datetime.now()

    date_str = str(date_str).strip()

    # Handle "yesterday"
    if 'yesterday' in date_str.lower():
        return reference_date - timedelta(days=1)

    # Handle "today" (in case it appears)
    if 'today' in date_str.lower():
        return reference_date

    # Extract date from "Added on DD/MM/YYYY" or "Reduced on DD/MM/YYYY"
    match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            return None

    return None


def get_event_type(date_str: str) -> str:
    """Determine if this is an 'Added' or 'Reduced' date."""
    if not date_str:
        return None
    date_str = str(date_str).lower()
    if 'added' in date_str:
        return 'added'
    elif 'reduced' in date_str:
        return 'reduced'
    return 'unknown'


def calculate_days_on_market(added_date: datetime, scraped_date: datetime) -> int:
    """Calculate days between added and scraped date."""
    if added_date is None or scraped_date is None:
        return None
    diff = scraped_date - added_date
    return max(0, diff.days)


def create_price_bucket(price_pcm: float) -> str:
    """Create price bucket labels."""
    if pd.isna(price_pcm) or price_pcm <= 0:
        return 'Unknown'
    elif price_pcm < 2000:
        return '< £2,000'
    elif price_pcm < 3000:
        return '£2,000-3,000'
    elif price_pcm < 5000:
        return '£3,000-5,000'
    elif price_pcm < 7500:
        return '£5,000-7,500'
    elif price_pcm < 10000:
        return '£7,500-10,000'
    elif price_pcm < 15000:
        return '£10,000-15,000'
    elif price_pcm < 25000:
        return '£15,000-25,000'
    else:
        return '£25,000+'


def load_and_process_data():
    """Load listings and process dates."""
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT
            source,
            property_id,
            address,
            postcode,
            area,
            price_pcm,
            bedrooms,
            size_sqft,
            added_date,
            scraped_at,
            let_agreed
        FROM listings
        WHERE price_pcm > 0
    """, conn)
    conn.close()

    print(f"Loaded {len(df):,} listings")

    # Parse scraped_at timestamp
    df['scraped_datetime'] = pd.to_datetime(df['scraped_at'], errors='coerce')

    # Use a consistent reference date (the most recent scrape date)
    reference_date = df['scraped_datetime'].max()
    print(f"Reference date (latest scrape): {reference_date}")

    # Parse added_date for Rightmove listings
    df['event_type'] = df['added_date'].apply(get_event_type)
    df['added_datetime'] = df['added_date'].apply(
        lambda x: parse_rightmove_date(x, reference_date)
    )

    # Calculate days on market
    df['days_on_market'] = df.apply(
        lambda row: calculate_days_on_market(row['added_datetime'], row['scraped_datetime']),
        axis=1
    )

    # Create price buckets
    df['price_bucket'] = df['price_pcm'].apply(create_price_bucket)

    # Order price buckets
    bucket_order = [
        '< £2,000', '£2,000-3,000', '£3,000-5,000', '£5,000-7,500',
        '£7,500-10,000', '£10,000-15,000', '£15,000-25,000', '£25,000+', 'Unknown'
    ]
    df['price_bucket'] = pd.Categorical(df['price_bucket'], categories=bucket_order, ordered=True)

    return df


def analyze_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze days on market by source."""
    # Only Rightmove has dates
    rightmove = df[df['source'] == 'rightmove'].copy()
    valid = rightmove[rightmove['days_on_market'].notna()]

    print(f"\n{'='*60}")
    print("DAYS ON MARKET ANALYSIS - BY SOURCE")
    print(f"{'='*60}")
    print(f"\nNote: Only Rightmove has added_date populated ({len(valid):,} listings)")
    print(f"Other sources ({', '.join(df[df['source']!='rightmove']['source'].unique())}) lack this data.\n")

    # Overall Rightmove stats
    stats = valid['days_on_market'].describe()
    print("Rightmove Overall Statistics:")
    print(f"  Count:     {stats['count']:,.0f}")
    print(f"  Mean:      {stats['mean']:.1f} days")
    print(f"  Median:    {stats['50%']:.1f} days")
    print(f"  Std Dev:   {stats['std']:.1f} days")
    print(f"  Min:       {stats['min']:.0f} days")
    print(f"  Max:       {stats['max']:.0f} days")
    print(f"  25th %ile: {stats['25%']:.1f} days")
    print(f"  75th %ile: {stats['75%']:.1f} days")

    # By event type (Added vs Reduced)
    print("\nBy Event Type (Added vs Reduced):")
    event_stats = valid.groupby('event_type')['days_on_market'].agg(['count', 'mean', 'median', 'std'])
    print(event_stats.to_string())

    return valid


def analyze_by_price_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze days on market by price bucket."""
    valid = df[(df['source'] == 'rightmove') & (df['days_on_market'].notna())].copy()

    print(f"\n{'='*60}")
    print("DAYS ON MARKET BY PRICE BUCKET")
    print(f"{'='*60}\n")

    stats = valid.groupby('price_bucket', observed=True)['days_on_market'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(1)
    stats.columns = ['Count', 'Mean Days', 'Median Days', 'Std Dev', 'Min', 'Max']

    print(stats.to_string())

    # Quick insights
    print("\nKey Insights:")
    fastest_bucket = stats['Median Days'].idxmin()
    slowest_bucket = stats['Median Days'].idxmax()
    print(f"  - Fastest to let: {fastest_bucket} ({stats.loc[fastest_bucket, 'Median Days']:.0f} day median)")
    print(f"  - Slowest to let: {slowest_bucket} ({stats.loc[slowest_bucket, 'Median Days']:.0f} day median)")

    return stats


def analyze_added_vs_reduced(df: pd.DataFrame):
    """Compare 'Added' vs 'Reduced' listings."""
    valid = df[(df['source'] == 'rightmove') & (df['days_on_market'].notna())].copy()

    print(f"\n{'='*60}")
    print("ADDED vs REDUCED PRICE LISTINGS")
    print(f"{'='*60}\n")

    added = valid[valid['event_type'] == 'added']
    reduced = valid[valid['event_type'] == 'reduced']

    print(f"Added listings:   {len(added):,} ({100*len(added)/len(valid):.1f}%)")
    print(f"Reduced listings: {len(reduced):,} ({100*len(reduced)/len(valid):.1f}%)")

    # Price comparison
    print(f"\nPrice comparison:")
    print(f"  Added median price:   £{added['price_pcm'].median():,.0f} pcm")
    print(f"  Reduced median price: £{reduced['price_pcm'].median():,.0f} pcm")

    # Days on market comparison
    print(f"\nDays on market (from listing date):")
    print(f"  Added median:   {added['days_on_market'].median():.0f} days")
    print(f"  Reduced median: {reduced['days_on_market'].median():.0f} days")

    # Reduced listings by price bucket
    print("\nPrice reductions by bucket:")
    reduced_by_bucket = valid.groupby('price_bucket', observed=True).apply(
        lambda x: pd.Series({
            'total': len(x),
            'reduced': (x['event_type'] == 'reduced').sum(),
            'pct_reduced': 100 * (x['event_type'] == 'reduced').sum() / len(x)
        })
    ).round(1)
    print(reduced_by_bucket.to_string())


def analyze_stale_listings(df: pd.DataFrame):
    """Identify potentially stale listings that have been on market a long time."""
    valid = df[(df['source'] == 'rightmove') & (df['days_on_market'].notna())].copy()

    print(f"\n{'='*60}")
    print("STALE LISTING ANALYSIS")
    print(f"{'='*60}\n")

    # Define stale thresholds
    thresholds = [7, 14, 21, 30, 45, 60, 90]

    print("Listings on market longer than threshold:")
    for days in thresholds:
        count = (valid['days_on_market'] >= days).sum()
        pct = 100 * count / len(valid)
        print(f"  >{days:2d} days: {count:,} listings ({pct:.1f}%)")

    # Long-term listings breakdown
    print("\nLong-term listings (>30 days) by price bucket:")
    long_term = valid[valid['days_on_market'] >= 30]
    lt_by_bucket = long_term.groupby('price_bucket', observed=True).agg({
        'days_on_market': ['count', 'mean', 'median']
    }).round(1)
    lt_by_bucket.columns = ['Count', 'Mean Days', 'Median Days']
    print(lt_by_bucket.to_string())

    # Show some example stale listings
    print("\nTop 10 longest listings on market:")
    top_stale = valid.nlargest(10, 'days_on_market')[
        ['address', 'postcode', 'price_pcm', 'bedrooms', 'days_on_market', 'event_type']
    ]
    for _, row in top_stale.iterrows():
        print(f"  {row['days_on_market']:.0f}d - £{row['price_pcm']:,.0f}/m - {row['bedrooms']}bed - {row['address'][:50]}")


def create_visualizations(df: pd.DataFrame):
    """Create comprehensive visualizations."""
    valid = df[(df['source'] == 'rightmove') & (df['days_on_market'].notna())].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('London Rental Market Duration Analysis\n(Rightmove Data)', fontsize=16, fontweight='bold')

    # 1. Distribution of days on market
    ax1 = axes[0, 0]
    valid['days_on_market'].hist(bins=50, ax=ax1, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(valid['days_on_market'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {valid['days_on_market'].median():.0f}d")
    ax1.axvline(valid['days_on_market'].mean(), color='orange', linestyle='--', linewidth=2, label=f"Mean: {valid['days_on_market'].mean():.0f}d")
    ax1.set_xlabel('Days on Market')
    ax1.set_ylabel('Number of Listings')
    ax1.set_title('Distribution of Days on Market')
    ax1.legend()
    ax1.set_xlim(0, 60)

    # 2. Days on market by price bucket (box plot)
    ax2 = axes[0, 1]
    bucket_order = ['< £2,000', '£2,000-3,000', '£3,000-5,000', '£5,000-7,500',
                    '£7,500-10,000', '£10,000-15,000', '£15,000-25,000', '£25,000+']
    plot_data = valid[valid['price_bucket'].isin(bucket_order)].copy()
    plot_data['price_bucket'] = pd.Categorical(plot_data['price_bucket'], categories=bucket_order, ordered=True)

    boxes = []
    labels = []
    for bucket in bucket_order:
        data = plot_data[plot_data['price_bucket'] == bucket]['days_on_market'].dropna()
        if len(data) > 0:
            boxes.append(data.values)
            labels.append(bucket.replace('£', '£\n').replace('-', '-\n'))

    bp = ax2.boxplot(boxes, labels=labels, patch_artist=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(boxes)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Days on Market')
    ax2.set_title('Days on Market by Price Bucket')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 60)

    # 3. Median days by price bucket (bar chart)
    ax3 = axes[0, 2]
    median_by_bucket = valid.groupby('price_bucket', observed=True)['days_on_market'].median()
    median_by_bucket = median_by_bucket.reindex(bucket_order).dropna()
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(median_by_bucket)))
    bars = ax3.bar(range(len(median_by_bucket)), median_by_bucket.values, color=colors, edgecolor='white')
    ax3.set_xticks(range(len(median_by_bucket)))
    ax3.set_xticklabels([b.replace('£', '£\n').replace('-', '-\n') for b in median_by_bucket.index], rotation=45, ha='right')
    ax3.set_ylabel('Median Days on Market')
    ax3.set_title('Median Days by Price Bucket')
    for bar, val in zip(bars, median_by_bucket.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}d',
                ha='center', va='bottom', fontsize=9)

    # 4. Added vs Reduced comparison
    ax4 = axes[1, 0]
    added = valid[valid['event_type'] == 'added']
    reduced = valid[valid['event_type'] == 'reduced']

    x = np.arange(2)
    width = 0.35
    counts = [len(added), len(reduced)]
    medians = [added['days_on_market'].median(), reduced['days_on_market'].median()]

    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(x - width/2, counts, width, label='Count', color='steelblue', alpha=0.7)
    bars2 = ax4_twin.bar(x + width/2, medians, width, label='Median Days', color='coral', alpha=0.7)

    ax4.set_xticks(x)
    ax4.set_xticklabels(['Added', 'Reduced'])
    ax4.set_ylabel('Number of Listings', color='steelblue')
    ax4_twin.set_ylabel('Median Days on Market', color='coral')
    ax4.set_title('Added vs Reduced Listings')

    # Add count labels
    for bar, val in zip(bars1, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val:,}',
                ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, medians):
        ax4_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}d',
                ha='center', va='bottom', fontsize=10)

    # 5. % Reduced by price bucket
    ax5 = axes[1, 1]
    reduced_pct = valid.groupby('price_bucket', observed=True).apply(
        lambda x: 100 * (x['event_type'] == 'reduced').sum() / len(x)
    )
    reduced_pct = reduced_pct.reindex(bucket_order).dropna()
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(reduced_pct)))
    bars = ax5.bar(range(len(reduced_pct)), reduced_pct.values, color=colors, edgecolor='white')
    ax5.set_xticks(range(len(reduced_pct)))
    ax5.set_xticklabels([b.replace('£', '£\n').replace('-', '-\n') for b in reduced_pct.index], rotation=45, ha='right')
    ax5.set_ylabel('% with Price Reduction')
    ax5.set_title('Price Reduction Rate by Bucket')
    ax5.axhline(reduced_pct.mean(), color='red', linestyle='--', label=f'Avg: {reduced_pct.mean():.1f}%')
    ax5.legend()
    for bar, val in zip(bars, reduced_pct.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}%',
                ha='center', va='bottom', fontsize=9)

    # 6. Cumulative distribution
    ax6 = axes[1, 2]
    sorted_days = np.sort(valid['days_on_market'].values)
    cumulative = np.arange(1, len(sorted_days) + 1) / len(sorted_days) * 100
    ax6.plot(sorted_days, cumulative, linewidth=2, color='steelblue')
    ax6.fill_between(sorted_days, cumulative, alpha=0.3, color='steelblue')

    # Add milestone lines
    for pct in [25, 50, 75, 90]:
        days_at_pct = np.percentile(valid['days_on_market'], pct)
        ax6.axhline(pct, color='gray', linestyle=':', alpha=0.5)
        ax6.axvline(days_at_pct, color='gray', linestyle=':', alpha=0.5)
        ax6.text(days_at_pct + 1, pct - 3, f'{pct}% by day {days_at_pct:.0f}', fontsize=9)

    ax6.set_xlabel('Days on Market')
    ax6.set_ylabel('Cumulative % of Listings')
    ax6.set_title('Cumulative Distribution')
    ax6.set_xlim(0, 60)
    ax6.set_ylim(0, 100)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'market_duration_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nVisualization saved to: {output_path}")

    return output_path


def create_heatmap(df: pd.DataFrame):
    """Create a heatmap of days on market by postcode and price bucket."""
    valid = df[(df['source'] == 'rightmove') & (df['days_on_market'].notna())].copy()

    # Get postcode district (first part)
    valid['postcode_district'] = valid['postcode'].str.extract(r'^([A-Z]+\d+)', expand=False)

    # Top 15 postcode districts by count
    top_postcodes = valid['postcode_district'].value_counts().head(15).index.tolist()

    bucket_order = ['< £2,000', '£2,000-3,000', '£3,000-5,000', '£5,000-7,500',
                    '£7,500-10,000', '£10,000-15,000', '£15,000-25,000', '£25,000+']

    # Create pivot table
    pivot = valid[valid['postcode_district'].isin(top_postcodes)].pivot_table(
        values='days_on_market',
        index='postcode_district',
        columns='price_bucket',
        aggfunc='median'
    )
    pivot = pivot.reindex(columns=[b for b in bucket_order if b in pivot.columns])

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=5, vmax=30)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                text_color = 'white' if val > 20 else 'black'
                ax.text(j, i, f'{val:.0f}d', ha='center', va='center',
                       fontsize=9, color=text_color)

    ax.set_xlabel('Price Bucket')
    ax.set_ylabel('Postcode District')
    ax.set_title('Median Days on Market by Postcode & Price\n(Rightmove Data)', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Median Days on Market')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'market_duration_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Heatmap saved to: {output_path}")

    return pivot


def main():
    """Run complete market duration analysis."""
    print("="*60)
    print("MARKET DURATION ANALYSIS")
    print("How long do London rental properties stay on market?")
    print("="*60)

    # Load and process data
    df = load_and_process_data()

    # Run analyses
    analyze_by_source(df)
    analyze_by_price_bucket(df)
    analyze_added_vs_reduced(df)
    analyze_stale_listings(df)

    # Create visualizations
    create_visualizations(df)
    create_heatmap(df)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
