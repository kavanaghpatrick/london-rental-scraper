#!/usr/bin/env python3
"""
Market Duration Analysis BY AGENCY

Analyzes how long properties stay on market by listing agent,
using Rightmove data which includes agent information.
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
    """
    Normalize agent names to parent company.
    E.g., "Foxtons, South Kensington" -> "Foxtons"
    """
    if not agent_name or pd.isna(agent_name):
        return 'Unknown'

    agent = str(agent_name).lower().strip()

    # Major agencies mapping
    agency_patterns = {
        'Foxtons': ['foxtons'],
        'Chestertons': ['chestertons'],
        'Knight Frank': ['knight frank'],
        'Savills': ['savills'],
        'Hamptons': ['hamptons'],
        'Marsh & Parsons': ['marsh & parsons', 'marsh and parsons'],
        'Dexters': ['dexters'],
        'John D Wood': ['john d wood'],
        'Winkworth': ['winkworth'],
        'Strutt & Parker': ['strutt & parker', 'strutt and parker'],
        'Carter Jonas': ['carter jonas'],
        'JLL': ['jll'],
        'CBRE': ['cbre'],
        'Cluttons': ['cluttons'],
        'Harrods Estates': ['harrods estates'],
        'Sothebys': ['sotheby'],
        'KFH': ['kfh', 'draker kfh'],
        'Domus Nova': ['domus nova'],
        'OpenRent': ['openrent'],
        'Hunters': ['hunters'],
        'Connells': ['connells'],
        'Barnard Marcus': ['barnard marcus'],
        'Felicity J Lord': ['felicity j lord'],
        'Benham & Reeves': ['benham & reeves', 'benham and reeves'],
        'Chancellors': ['chancellors'],
        'Portico': ['portico'],
        'Lurot Brand': ['lurot brand'],
        'Russell Simpson': ['russell simpson'],
        'Aylesford': ['aylesford'],
        'Arlington Residential': ['arlington residential'],
        'Aston Chase': ['aston chase'],
        'Beauchamp Estates': ['beauchamp'],
        'W.A. Ellis': ['w.a. ellis', 'wa ellis'],
        'Cluttons': ['cluttons'],
        'GPE': ['gpe'],
        'Landlord Sales Agency': ['landlord sales'],
        'RR Properties': ['rr properties'],
        'City Relay': ['city relay'],
        'Landstones': ['landstones'],
        'Eson2': ['eson2'],
    }

    for agency, patterns in agency_patterns.items():
        for pattern in patterns:
            if pattern in agent:
                return agency

    # If no match, extract first part before comma
    if ',' in agent_name:
        return agent_name.split(',')[0].strip()

    return agent_name[:30] if len(agent_name) > 30 else agent_name


def create_price_bucket(price_pcm: float) -> str:
    """Create price bucket labels."""
    if pd.isna(price_pcm) or price_pcm <= 0:
        return 'Unknown'
    elif price_pcm < 2000:
        return '< £2k'
    elif price_pcm < 3000:
        return '£2-3k'
    elif price_pcm < 5000:
        return '£3-5k'
    elif price_pcm < 7500:
        return '£5-7.5k'
    elif price_pcm < 10000:
        return '£7.5-10k'
    elif price_pcm < 15000:
        return '£10-15k'
    elif price_pcm < 25000:
        return '£15-25k'
    else:
        return '£25k+'


def load_and_process_data():
    """Load Rightmove listings with agent data."""
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT
            property_id,
            address,
            postcode,
            area,
            price_pcm,
            bedrooms,
            size_sqft,
            added_date,
            scraped_at,
            agent_name
        FROM listings
        WHERE source = 'rightmove'
          AND price_pcm > 0
          AND agent_name IS NOT NULL
          AND agent_name != ''
    """, conn)
    conn.close()

    print(f"Loaded {len(df):,} Rightmove listings with agent data")

    # Parse dates
    df['scraped_datetime'] = pd.to_datetime(df['scraped_at'], errors='coerce')
    reference_date = df['scraped_datetime'].max()

    df['added_datetime'] = df['added_date'].apply(
        lambda x: parse_rightmove_date(x, reference_date)
    )

    # Calculate days on market
    df['days_on_market'] = df.apply(
        lambda row: max(0, (row['scraped_datetime'] - row['added_datetime']).days)
        if row['added_datetime'] and row['scraped_datetime'] else None,
        axis=1
    )

    # Normalize agency names
    df['agency'] = df['agent_name'].apply(normalize_agency)

    # Create price buckets
    df['price_bucket'] = df['price_pcm'].apply(create_price_bucket)

    return df


def analyze_by_agency(df: pd.DataFrame):
    """Analyze days on market by agency."""
    valid = df[df['days_on_market'].notna()].copy()

    print(f"\n{'='*70}")
    print("DAYS ON MARKET BY AGENCY")
    print(f"{'='*70}\n")

    # Get agencies with at least 20 listings
    agency_counts = valid['agency'].value_counts()
    major_agencies = agency_counts[agency_counts >= 20].index.tolist()

    print(f"Agencies with 20+ listings: {len(major_agencies)}")

    # Calculate stats per agency
    stats = valid[valid['agency'].isin(major_agencies)].groupby('agency').agg({
        'days_on_market': ['count', 'mean', 'median', 'std'],
        'price_pcm': 'median'
    }).round(1)
    stats.columns = ['Count', 'Mean Days', 'Median Days', 'Std Dev', 'Median Price']
    stats = stats.sort_values('Median Days')

    print("\nAgencies ranked by FASTEST to let (lowest median days):")
    print(stats.to_string())

    # Top 5 fastest and slowest
    print(f"\n{'='*50}")
    print("TOP 5 FASTEST AGENCIES (lowest median days):")
    for i, (agency, row) in enumerate(stats.head(5).iterrows(), 1):
        print(f"  {i}. {agency}: {row['Median Days']:.0f} days median ({row['Count']:.0f} listings, £{row['Median Price']:,.0f} median price)")

    print(f"\nTOP 5 SLOWEST AGENCIES (highest median days):")
    for i, (agency, row) in enumerate(stats.tail(5).iloc[::-1].iterrows(), 1):
        print(f"  {i}. {agency}: {row['Median Days']:.0f} days median ({row['Count']:.0f} listings, £{row['Median Price']:,.0f} median price)")

    return stats


def analyze_agency_by_price(df: pd.DataFrame):
    """Cross-tabulation of agency × price bucket."""
    valid = df[df['days_on_market'].notna()].copy()

    # Get top agencies
    top_agencies = valid['agency'].value_counts().head(15).index.tolist()

    bucket_order = ['< £2k', '£2-3k', '£3-5k', '£5-7.5k', '£7.5-10k', '£10-15k', '£15-25k', '£25k+']

    print(f"\n{'='*70}")
    print("MEDIAN DAYS ON MARKET: AGENCY × PRICE BUCKET")
    print(f"{'='*70}\n")

    # Create pivot table
    pivot = valid[valid['agency'].isin(top_agencies)].pivot_table(
        values='days_on_market',
        index='agency',
        columns='price_bucket',
        aggfunc='median'
    )
    pivot = pivot.reindex(columns=[b for b in bucket_order if b in pivot.columns])

    # Sort by overall median
    agency_medians = valid[valid['agency'].isin(top_agencies)].groupby('agency')['days_on_market'].median()
    pivot = pivot.reindex(agency_medians.sort_values().index)

    print(pivot.round(0).to_string())

    return pivot


def create_agency_visualizations(df: pd.DataFrame):
    """Create visualizations for agency analysis."""
    valid = df[df['days_on_market'].notna()].copy()

    # Get top 15 agencies by listing count
    top_agencies = valid['agency'].value_counts().head(15).index.tolist()
    plot_data = valid[valid['agency'].isin(top_agencies)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Market Duration by Agency\n(Rightmove Data - Listings with Agent Info)',
                 fontsize=16, fontweight='bold')

    # 1. Median days by agency (bar chart, sorted)
    ax1 = axes[0, 0]
    agency_stats = plot_data.groupby('agency')['days_on_market'].agg(['median', 'count'])
    agency_stats = agency_stats.sort_values('median')

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(agency_stats)))
    bars = ax1.barh(range(len(agency_stats)), agency_stats['median'].values, color=colors)
    ax1.set_yticks(range(len(agency_stats)))
    ax1.set_yticklabels(agency_stats.index)
    ax1.set_xlabel('Median Days on Market')
    ax1.set_title('Agencies Ranked by Speed to Let')

    # Add count labels
    for i, (bar, (idx, row)) in enumerate(zip(bars, agency_stats.iterrows())):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{row["median"]:.0f}d (n={row["count"]:.0f})',
                va='center', fontsize=9)
    ax1.set_xlim(0, agency_stats['median'].max() * 1.3)

    # 2. Box plot by agency
    ax2 = axes[0, 1]
    agency_order = agency_stats.index.tolist()
    boxes = [plot_data[plot_data['agency'] == a]['days_on_market'].values for a in agency_order]

    bp = ax2.boxplot(boxes, vert=False, patch_artist=True, tick_labels=agency_order)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xlabel('Days on Market')
    ax2.set_title('Distribution of Days on Market by Agency')
    ax2.set_xlim(0, 60)

    # 3. Heatmap: Agency × Price bucket
    ax3 = axes[1, 0]
    bucket_order = ['< £2k', '£2-3k', '£3-5k', '£5-7.5k', '£7.5-10k', '£10-15k', '£15-25k', '£25k+']

    pivot = plot_data.pivot_table(
        values='days_on_market',
        index='agency',
        columns='price_bucket',
        aggfunc='median'
    )
    pivot = pivot.reindex(columns=[b for b in bucket_order if b in pivot.columns])
    pivot = pivot.reindex(agency_order)

    im = ax3.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=5, vmax=35)
    ax3.set_xticks(np.arange(len(pivot.columns)))
    ax3.set_yticks(np.arange(len(pivot.index)))
    ax3.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax3.set_yticklabels(pivot.index)
    ax3.set_title('Median Days: Agency × Price Bucket')

    # Add text
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                text_color = 'white' if val > 25 else 'black'
                ax3.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=8, color=text_color)

    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Median Days')

    # 4. Scatter: Median Price vs Median Days (per agency)
    ax4 = axes[1, 1]
    agency_summary = plot_data.groupby('agency').agg({
        'days_on_market': 'median',
        'price_pcm': 'median',
        'property_id': 'count'
    }).rename(columns={'property_id': 'count'})

    sizes = agency_summary['count'] * 3
    scatter = ax4.scatter(agency_summary['price_pcm'], agency_summary['days_on_market'],
                         s=sizes, c=agency_summary['days_on_market'], cmap='RdYlGn_r',
                         alpha=0.7, edgecolors='black', linewidth=0.5)

    # Label points
    for agency, row in agency_summary.iterrows():
        ax4.annotate(agency, (row['price_pcm'], row['days_on_market']),
                    fontsize=8, ha='left', va='bottom',
                    xytext=(5, 5), textcoords='offset points')

    ax4.set_xlabel('Median Listing Price (£/month)')
    ax4.set_ylabel('Median Days on Market')
    ax4.set_title('Agency Position: Price vs Speed\n(bubble size = listing count)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'market_duration_by_agency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nVisualization saved to: {output_path}")

    return output_path


def price_reduction_by_agency(df: pd.DataFrame):
    """Analyze price reduction rates by agency."""
    valid = df.copy()
    valid['was_reduced'] = valid['added_date'].str.lower().str.contains('reduced', na=False)

    # Top agencies
    top_agencies = valid['agency'].value_counts().head(15).index.tolist()

    print(f"\n{'='*70}")
    print("PRICE REDUCTION RATE BY AGENCY")
    print(f"{'='*70}\n")

    reduction_stats = valid[valid['agency'].isin(top_agencies)].groupby('agency').agg({
        'was_reduced': ['sum', 'count', 'mean']
    })
    reduction_stats.columns = ['Reduced', 'Total', 'Reduction Rate']
    reduction_stats['Reduction Rate'] = (reduction_stats['Reduction Rate'] * 100).round(1)
    reduction_stats = reduction_stats.sort_values('Reduction Rate')

    print("Agencies ranked by LOWEST price reduction rate:")
    print(reduction_stats.to_string())

    return reduction_stats


def main():
    """Run agency-specific market duration analysis."""
    print("="*70)
    print("MARKET DURATION ANALYSIS BY AGENCY")
    print("="*70)

    df = load_and_process_data()

    agency_stats = analyze_by_agency(df)
    pivot = analyze_agency_by_price(df)
    reduction_stats = price_reduction_by_agency(df)

    create_agency_visualizations(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
