#!/usr/bin/env python3
"""
SW1 Belgravia/Westminster Analysis: Chestertons vs Competition

Focused comparison in Chestertons' key market.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'output', 'rentals.db')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def load_sw1_data():
    """Load SW1 data from all sources."""
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT
            source,
            property_id,
            address,
            postcode,
            price_pcm,
            bedrooms,
            size_sqft,
            property_type
        FROM listings
        WHERE price_pcm > 0
          AND postcode LIKE 'SW1%'
          AND source IN ('chestertons', 'savills', 'knightfrank')
    """, conn)
    conn.close()

    # Normalize agency names
    df['agency'] = df['source'].map({
        'chestertons': 'Chestertons',
        'savills': 'Savills',
        'knightfrank': 'Knight Frank'
    })

    # Calculate price per sqft
    df['ppsf'] = df.apply(
        lambda r: r['price_pcm'] / r['size_sqft'] if r['size_sqft'] and r['size_sqft'] > 0 else None,
        axis=1
    )

    # Price buckets
    df['price_bucket'] = pd.cut(
        df['price_pcm'],
        bins=[0, 3000, 5000, 7500, 10000, 15000, float('inf')],
        labels=['<£3k', '£3-5k', '£5-7.5k', '£7.5-10k', '£10-15k', '£15k+']
    )

    return df


def create_sw1_visualization(df: pd.DataFrame):
    """Create SW1 comparison visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SW1 (Belgravia/Westminster) Market Analysis\nChestertons vs Competition',
                 fontsize=16, fontweight='bold')

    colors = {
        'Chestertons': '#2E86AB',
        'Savills': '#A23B72',
        'Knight Frank': '#F18F01'
    }

    agencies = ['Chestertons', 'Savills', 'Knight Frank']

    # 1. Market Share (listing count)
    ax1 = axes[0, 0]
    counts = df['agency'].value_counts().reindex(agencies)
    bars = ax1.bar(agencies, counts.values, color=[colors[a] for a in agencies], edgecolor='white')
    ax1.set_ylabel('Number of Listings')
    ax1.set_title('Market Presence in SW1')
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val}\n({100*val/counts.sum():.0f}%)', ha='center', va='bottom', fontsize=10)

    # 2. Price Distribution (box plot)
    ax2 = axes[0, 1]
    box_data = [df[df['agency'] == a]['price_pcm'].values for a in agencies]
    bp = ax2.boxplot(box_data, tick_labels=agencies, patch_artist=True)
    for patch, agency in zip(bp['boxes'], agencies):
        patch.set_facecolor(colors[agency])
        patch.set_alpha(0.7)
    ax2.set_ylabel('Monthly Rent (£)')
    ax2.set_title('Price Distribution')
    ax2.set_ylim(0, 25000)

    # Add medians
    for i, agency in enumerate(agencies):
        med = df[df['agency'] == agency]['price_pcm'].median()
        ax2.text(i + 1, med + 500, f'£{med:,.0f}', ha='center', fontsize=9, fontweight='bold')

    # 3. Price per Sqft Comparison
    ax3 = axes[0, 2]
    df_sqft = df[df['ppsf'].notna() & (df['ppsf'] < 15)]

    for agency in agencies:
        data = df_sqft[df_sqft['agency'] == agency]['ppsf']
        ax3.hist(data, bins=20, alpha=0.5, label=f'{agency} (med: £{data.median():.2f})',
                color=colors[agency], density=True)

    ax3.set_xlabel('Price per Sqft (£)')
    ax3.set_ylabel('Density')
    ax3.set_title('Value Comparison (£/sqft)')
    ax3.legend()

    # 4. Portfolio by Price Segment
    ax4 = axes[1, 0]
    bucket_order = ['<£3k', '£3-5k', '£5-7.5k', '£7.5-10k', '£10-15k', '£15k+']

    x = np.arange(len(bucket_order))
    width = 0.25

    for i, agency in enumerate(agencies):
        pcts = []
        agency_data = df[df['agency'] == agency]
        for bucket in bucket_order:
            cnt = len(agency_data[agency_data['price_bucket'] == bucket])
            pcts.append(100 * cnt / len(agency_data))
        ax4.bar(x + i * width, pcts, width, label=agency, color=colors[agency], alpha=0.8)

    ax4.set_xticks(x + width)
    ax4.set_xticklabels(bucket_order, rotation=45, ha='right')
    ax4.set_ylabel('% of Portfolio')
    ax4.set_title('Price Segment Focus')
    ax4.legend()

    # 5. Average Price by Bedroom
    ax5 = axes[1, 1]
    beds_range = [1, 2, 3, 4, 5]
    x = np.arange(len(beds_range))
    width = 0.25

    for i, agency in enumerate(agencies):
        prices = []
        for beds in beds_range:
            data = df[(df['agency'] == agency) & (df['bedrooms'] == beds)]
            prices.append(data['price_pcm'].median() if len(data) > 0 else 0)
        ax5.bar(x + i * width, prices, width, label=agency, color=colors[agency], alpha=0.8)

    ax5.set_xticks(x + width)
    ax5.set_xticklabels([f'{b} bed' for b in beds_range])
    ax5.set_ylabel('Median Price (£/month)')
    ax5.set_title('Pricing by Bedroom Count')
    ax5.legend()

    # 6. Price per Sqft by Bedroom
    ax6 = axes[1, 2]

    for i, agency in enumerate(agencies):
        ppsf_vals = []
        for beds in beds_range:
            data = df[(df['agency'] == agency) & (df['bedrooms'] == beds) & (df['ppsf'].notna())]
            ppsf_vals.append(data['ppsf'].median() if len(data) > 0 else 0)
        ax6.bar(x + i * width, ppsf_vals, width, label=agency, color=colors[agency], alpha=0.8)

    ax6.set_xticks(x + width)
    ax6.set_xticklabels([f'{b} bed' for b in beds_range])
    ax6.set_ylabel('Median £/sqft')
    ax6.set_title('Value by Bedroom Count')
    ax6.legend()

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'sw1_chestertons_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

    return output_path


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""

    print("=" * 70)
    print(" SW1 MARKET ANALYSIS: CHESTERTONS vs COMPETITION")
    print("=" * 70)

    agencies = ['Chestertons', 'Savills', 'Knight Frank']

    print(f"\n{'Agency':<15} {'Listings':>10} {'Share':>8} {'Med Price':>12} {'Med Sqft':>10} {'Med £/sqft':>12}")
    print("-" * 70)

    total = len(df)
    for agency in agencies:
        data = df[df['agency'] == agency]
        sqft_data = data[data['size_sqft'] > 0]

        count = len(data)
        share = 100 * count / total
        med_price = data['price_pcm'].median()
        med_sqft = sqft_data['size_sqft'].median() if len(sqft_data) > 0 else 0
        med_ppsf = sqft_data['ppsf'].median() if len(sqft_data) > 0 else 0

        print(f"{agency:<15} {count:>10,} {share:>7.1f}% £{med_price:>10,.0f} {med_sqft:>10,.0f} £{med_ppsf:>11.2f}")

    # Chestertons vs Savills comparison (main competitors)
    chester = df[df['agency'] == 'Chestertons']
    savills = df[df['agency'] == 'Savills']

    chester_ppsf = chester[chester['ppsf'].notna()]['ppsf'].median()
    savills_ppsf = savills[savills['ppsf'].notna()]['ppsf'].median()

    print("\n" + "=" * 70)
    print(" KEY FINDING: CHESTERTONS vs SAVILLS VALUE COMPARISON")
    print("=" * 70)
    print(f"\n  Chestertons median £/sqft: £{chester_ppsf:.2f}")
    print(f"  Savills median £/sqft:     £{savills_ppsf:.2f}")
    print(f"  Difference:                {100*(chester_ppsf/savills_ppsf - 1):+.1f}%")

    if chester_ppsf > savills_ppsf:
        print(f"\n  → Chestertons charges {100*(chester_ppsf/savills_ppsf - 1):.0f}% MORE per sqft than Savills")
    else:
        print(f"\n  → Chestertons charges {100*(1 - chester_ppsf/savills_ppsf):.0f}% LESS per sqft than Savills")

    # Breakdown by bedroom
    print("\n" + "-" * 70)
    print(" £/SQFT BY BEDROOM COUNT")
    print("-" * 70)
    print(f"{'Beds':<8} {'Chestertons':>15} {'Savills':>15} {'Knight Frank':>15} {'C vs S':>12}")

    for beds in [1, 2, 3, 4, 5]:
        c_data = chester[(chester['bedrooms'] == beds) & (chester['ppsf'].notna())]
        s_data = savills[(savills['bedrooms'] == beds) & (savills['ppsf'].notna())]
        kf_data = df[(df['agency'] == 'Knight Frank') & (df['bedrooms'] == beds) & (df['ppsf'].notna())]

        c_ppsf = c_data['ppsf'].median() if len(c_data) > 0 else None
        s_ppsf = s_data['ppsf'].median() if len(s_data) > 0 else None
        kf_ppsf = kf_data['ppsf'].median() if len(kf_data) > 0 else None

        c_str = f"£{c_ppsf:.2f}" if c_ppsf else "N/A"
        s_str = f"£{s_ppsf:.2f}" if s_ppsf else "N/A"
        kf_str = f"£{kf_ppsf:.2f}" if kf_ppsf else "N/A"

        if c_ppsf and s_ppsf:
            diff = 100 * (c_ppsf / s_ppsf - 1)
            diff_str = f"{diff:+.0f}%"
        else:
            diff_str = "N/A"

        print(f"{beds} bed   {c_str:>15} {s_str:>15} {kf_str:>15} {diff_str:>12}")


def main():
    df = load_sw1_data()
    print_summary(df)
    create_sw1_visualization(df)


if __name__ == '__main__':
    main()
