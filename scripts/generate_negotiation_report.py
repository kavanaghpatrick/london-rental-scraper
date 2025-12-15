#!/usr/bin/env python3
"""
Generate a data-driven property valuation report with charts.

Usage:
    python scripts/generate_negotiation_report.py \
        --address "4 South Eaton Place" \
        --postcode SW1W \
        --size 1312 \
        --beds 2 \
        --baths 2 \
        --predicted 8925 \
        --low 7318 \
        --high 10531

Output:
    output/negotiation_report/negotiation_report.html
    output/negotiation_report/chart*.png
    output/negotiation_report/comparables.csv
    output/negotiation_report/statistics.json
"""

import argparse
import sqlite3
import json
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'output' / 'rentals.db'
OUTPUT_DIR = PROJECT_DIR / 'output' / 'negotiation_report'

# Database connection - supports Postgres (if POSTGRES_URL set) or SQLite
def get_db_connection():
    """Get database connection - Postgres if POSTGRES_URL set, else SQLite."""
    postgres_url = os.environ.get('POSTGRES_URL')
    if postgres_url:
        import psycopg2
        return psycopg2.connect(postgres_url), 'postgres'
    else:
        return sqlite3.connect(DB_PATH), 'sqlite'

# SW1 area districts (NOT SW10, SW11, SW12 - those are different areas!)
SW1_DISTRICTS = {'SW1A', 'SW1E', 'SW1H', 'SW1P', 'SW1V', 'SW1W', 'SW1X', 'SW1Y'}

# Prime adjacent areas for SW1
PRIME_ADJACENT = {'SW3', 'SW7', 'W1', 'W8', 'NW1', 'NW3', 'NW8'}

# Subject property (defaults, can be overridden by CLI)
SUBJECT = {
    'address': '4 South Eaton Place',
    'postcode': 'SW1W',
    'size_sqft': 1312,
    'bedrooms': 2,
    'bathrooms': 2,
    'predicted_pcm': 8925,  # Model prediction
    'range_low': 7318,
    'range_high': 10531,
}


def extract_district(postcode):
    """
    Extract outward code (district) from postcode.
    'SW1W 9AB' -> 'SW1W'
    'SW11 4AP' -> 'SW11'
    'W1' -> 'W1'
    """
    if not postcode or pd.isna(postcode):
        return ''
    postcode = str(postcode).upper().strip()
    # Split on space and take first part
    parts = postcode.split()
    return parts[0] if parts else postcode


def is_sw1_district(district):
    """
    Check if a district is in the SW1 area (SW1A-SW1Y).
    NOT SW10, SW11, SW12 etc. - those are different areas!
    """
    if not district:
        return False
    # SW1 districts are SW1 followed by a letter (A-Y)
    # SW10, SW11 etc have SW1 followed by a digit
    return district in SW1_DISTRICTS or (
        district.startswith('SW1') and
        len(district) == 4 and
        district[3].isalpha()
    )


def get_area_code(district):
    """
    Extract area code from district.
    'SW1W' -> 'SW1' (for SW1 area districts)
    'SW11' -> 'SW11' (different area - not SW1!)
    'W1K' -> 'W1'
    'NW3' -> 'NW3'
    """
    if not district:
        return ''

    # Match pattern: letters followed by digits, optionally followed by more chars
    match = re.match(r'^([A-Z]+)(\d+)', district)
    if not match:
        return district

    letters, digits = match.groups()

    # Check if this is SW1 area (letter after digit) vs SW10/SW11 (digit after digit)
    if len(district) > len(letters) + len(digits):
        next_char = district[len(letters) + len(digits)]
        if next_char.isalpha():
            # SW1W format -> area is SW1 (or EC1A -> EC1)
            return letters + digits[0] if len(digits) == 1 else letters + digits

    return letters + digits


def get_comp_tier(district, subject_district):
    """
    Determine the tier for a comparable based on location.

    Tier 1: Same district (e.g., SW1W = SW1W)
    Tier 2: Same area (e.g., SW1W ↔ SW1X, both in SW1)
    Tier 3: Adjacent prime (e.g., SW1 ↔ SW3, SW7, W1)
    Tier 4: Broader market
    """
    if not district or not subject_district:
        return 4, 'Tier 4: Broader Market'

    district = district.upper()
    subject_district = subject_district.upper()

    # Tier 1: Exact match
    if district == subject_district:
        return 1, 'Tier 1: Same District'

    # For SW1 subjects
    if is_sw1_district(subject_district):
        # Tier 2: Other SW1 districts
        if is_sw1_district(district):
            return 2, 'Tier 2: SW1 Area'

        # Tier 3: Adjacent prime areas
        area = get_area_code(district)
        if area in PRIME_ADJACENT or district in PRIME_ADJACENT:
            return 3, 'Tier 3: Prime Central'

    # For other prime subjects
    subject_area = get_area_code(subject_district)
    comp_area = get_area_code(district)

    # Tier 2: Same area
    if comp_area == subject_area:
        return 2, f'Tier 2: {subject_area} Area'

    # Tier 3: Adjacent prime
    if comp_area in PRIME_ADJACENT or district in PRIME_ADJACENT:
        return 3, 'Tier 3: Prime Central'

    return 4, 'Tier 4: Broader Market'


def load_comparables(subject, size_range=0.20, bed_match=True):
    """
    Load and score comparable properties from database.

    Scoring system:
    - Location (50%): Same district (100), Same area (80), Prime adjacent (50), Other (20)
    - Size similarity (25%): Linear scale based on % difference
    - PPSF similarity (25%): Linear scale based on % difference
    """
    conn, db_type = get_db_connection()

    min_size = subject['size_sqft'] * (1 - size_range)
    max_size = subject['size_sqft'] * (1 + size_range)

    # Use appropriate placeholder for database type
    ph = '%s' if db_type == 'postgres' else '?'

    query = f"""
    SELECT
        address, postcode, source, price_pcm, size_sqft,
        bedrooms, bathrooms, url,
        CAST(price_pcm AS FLOAT) / size_sqft as ppsf
    FROM listings
    WHERE is_active = 1
      AND size_sqft IS NOT NULL
      AND size_sqft > 0
      AND price_pcm IS NOT NULL
      AND price_pcm > 0
      AND size_sqft BETWEEN {ph} AND {ph}
    """
    params = [min_size, max_size]

    if bed_match:
        query += f" AND bedrooms = {ph}"
        params.append(subject['bedrooms'])

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if len(df) == 0:
        return df

    # Extract subject info
    subject_district = extract_district(subject['postcode'])
    subject_ppsf = subject['predicted_pcm'] / subject['size_sqft']

    # Add district column
    df['district'] = df['postcode'].apply(extract_district)

    # Calculate tier and score for each comparable
    def score_comparable(row):
        district = row['district']

        # Location score (0-100, weight 50%)
        tier_num, tier_label = get_comp_tier(district, subject_district)
        location_scores = {1: 100, 2: 80, 3: 50, 4: 20}
        location_score = location_scores.get(tier_num, 20)

        # Size similarity score (0-50, weight 25%)
        size_diff_pct = abs(row['size_sqft'] - subject['size_sqft']) / subject['size_sqft']
        size_score = max(0, 50 * (1 - size_diff_pct / size_range))

        # PPSF similarity score (0-50, weight 25%)
        ppsf_diff_pct = abs(row['ppsf'] - subject_ppsf) / subject_ppsf
        ppsf_score = max(0, 50 * (1 - min(ppsf_diff_pct, 0.5) / 0.5))

        total_score = (location_score * 0.5) + (size_score * 0.5) + (ppsf_score * 0.5)

        return pd.Series({
            'tier_num': tier_num,
            'tier': tier_label,
            'location_score': location_score,
            'size_score': size_score,
            'ppsf_score': ppsf_score,
            'comp_score': total_score
        })

    # Apply scoring
    scores = df.apply(score_comparable, axis=1)
    df = pd.concat([df, scores], axis=1)

    # Sort by score (highest first)
    df = df.sort_values('comp_score', ascending=False)

    return df


def load_all_listings_with_ppsf():
    """Load all listings with valid PPSF for distribution analysis."""
    conn, db_type = get_db_connection()

    # Use database-specific substring function
    if db_type == 'postgres':
        district_expr = """
            CASE
                WHEN POSITION(' ' IN postcode) > 0 THEN SUBSTRING(postcode, 1, POSITION(' ' IN postcode) - 1)
                ELSE postcode
            END
        """
    else:
        district_expr = """
            SUBSTR(postcode, 1,
                CASE
                    WHEN INSTR(postcode, ' ') > 0 THEN INSTR(postcode, ' ') - 1
                    ELSE LENGTH(postcode)
                END
            )
        """

    query = f"""
    SELECT
        address, postcode, source, price_pcm, size_sqft, bedrooms,
        CAST(price_pcm AS FLOAT) / size_sqft as ppsf,
        {district_expr} as district
    FROM listings
    WHERE is_active = 1
      AND size_sqft IS NOT NULL AND size_sqft > 100
      AND price_pcm IS NOT NULL AND price_pcm > 500
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def calc_statistics(comps, all_listings, subject):
    """Calculate statistics from comparables including tier breakdowns."""
    ppsf_values = comps['ppsf'].dropna()
    all_ppsf = all_listings['ppsf'].dropna()
    subject_ppsf = subject['predicted_pcm'] / subject['size_sqft']

    # Get SW1 specific data - use proper SW1 detection, not string matching!
    if 'district' in comps.columns:
        sw1_comps = comps[comps['district'].apply(lambda x: is_sw1_district(str(x)) if pd.notna(x) else False)]
    else:
        sw1_comps = pd.DataFrame()

    # For all_listings, add district column if not present
    if 'district' not in all_listings.columns:
        all_listings = all_listings.copy()
        all_listings['district'] = all_listings['postcode'].apply(extract_district)

    sw1_all = all_listings[all_listings['district'].apply(lambda x: is_sw1_district(str(x)) if pd.notna(x) else False)]

    # Tier breakdowns
    tier_stats = {}
    if 'tier_num' in comps.columns:
        for tier_num in [1, 2, 3, 4]:
            tier_comps = comps[comps['tier_num'] == tier_num]
            if len(tier_comps) > 0:
                tier_stats[f'tier{tier_num}_count'] = len(tier_comps)
                tier_stats[f'tier{tier_num}_median_ppsf'] = round(tier_comps['ppsf'].median(), 2)
                tier_stats[f'tier{tier_num}_median_rent'] = round(tier_comps['price_pcm'].median(), 0)
            else:
                tier_stats[f'tier{tier_num}_count'] = 0
                tier_stats[f'tier{tier_num}_median_ppsf'] = None
                tier_stats[f'tier{tier_num}_median_rent'] = None

    stats = {
        'total_comps': len(comps),
        'total_market': len(all_listings),
        'sw1_comps': len(sw1_comps),
        'sw1_all': len(sw1_all),
        'median_ppsf': round(ppsf_values.median(), 2) if len(ppsf_values) > 0 else None,
        'mean_ppsf': round(ppsf_values.mean(), 2) if len(ppsf_values) > 0 else None,
        'p25_ppsf': round(ppsf_values.quantile(0.25), 2) if len(ppsf_values) > 0 else None,
        'p75_ppsf': round(ppsf_values.quantile(0.75), 2) if len(ppsf_values) > 0 else None,
        'subject_ppsf': round(subject_ppsf, 2),
        'market_percentile': round(
            (all_ppsf < subject_ppsf).sum() / len(all_ppsf) * 100, 1
        ) if len(all_ppsf) > 0 else None,
        'comp_percentile': round(
            (ppsf_values < subject_ppsf).sum() / len(ppsf_values) * 100, 1
        ) if len(ppsf_values) > 0 else None,
        'sw1_median_ppsf': round(sw1_all['ppsf'].median(), 2) if len(sw1_all) > 0 else None,
        'sw1_median_rent': round(sw1_comps['price_pcm'].median(), 0) if len(sw1_comps) > 0 else None,
        'market_median_rent': round(comps['price_pcm'].median(), 0) if len(comps) > 0 else None,
        **tier_stats,
    }
    return stats


def chart_ppsf_distribution(all_listings, subject, output_path):
    """Chart 1: PPSF distribution with subject marker."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ppsf = all_listings['ppsf'].dropna()
    ppsf = ppsf[(ppsf > 1) & (ppsf < 25)]  # Filter outliers

    subject_ppsf = subject['predicted_pcm'] / subject['size_sqft']

    ax.hist(ppsf, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(subject_ppsf, color='red', linewidth=2, linestyle='--',
               label=f'Subject: £{subject_ppsf:.2f}/sqft')
    ax.axvline(ppsf.median(), color='green', linewidth=2, linestyle='-.',
               label=f'Market Median: £{ppsf.median():.2f}/sqft')

    ax.set_xlabel('Price per Square Foot (£/sqft)', fontsize=12)
    ax.set_ylabel('Number of Properties', fontsize=12)
    ax.set_title('Price per Sqft Distribution - London Rentals', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('£{x:.0f}'))

    percentile = (ppsf < subject_ppsf).sum() / len(ppsf) * 100
    ax.annotate(f'{percentile:.0f}th percentile',
                xy=(subject_ppsf, ax.get_ylim()[1] * 0.8),
                fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def chart_size_vs_price(comps, subject, output_path):
    """Chart 2: Size vs Price scatter with subject."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by source
    sources = comps['source'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(sources)))

    for source, color in zip(sources, colors):
        mask = comps['source'] == source
        ax.scatter(comps[mask]['size_sqft'], comps[mask]['price_pcm'],
                   alpha=0.6, s=50, c=[color], label=source, edgecolors='white')

    # Subject property
    ax.scatter(subject['size_sqft'], subject['predicted_pcm'],
               s=200, c='red', marker='*', edgecolors='black', linewidth=2,
               label=f"Subject (£{subject['predicted_pcm']:,})", zorder=10)

    # Confidence range
    ax.fill_between([subject['size_sqft'] - 20, subject['size_sqft'] + 20],
                    subject['range_low'], subject['range_high'],
                    alpha=0.2, color='red', label='Confidence Range')

    ax.set_xlabel('Size (sqft)', fontsize=12)
    ax.set_ylabel('Rent (£ pcm)', fontsize=12)
    ax.set_title('Size vs Monthly Rent - Comparable Properties', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('£{x:,.0f}'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def chart_price_by_district(all_listings, subject, output_path):
    """Chart 3: Box plot of PPSF by district."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get top districts by count
    district_counts = all_listings['district'].value_counts()
    top_districts = district_counts[district_counts >= 5].index[:15].tolist()

    # Ensure subject district is included - use extract_district for proper parsing
    subject_district = extract_district(subject['postcode'])
    if subject_district not in top_districts:
        top_districts.append(subject_district)

    df_plot = all_listings[all_listings['district'].isin(top_districts)].copy()

    # Sort by median PPSF
    district_medians = df_plot.groupby('district')['ppsf'].median().sort_values(ascending=False)

    # Box plot
    data = [df_plot[df_plot['district'] == d]['ppsf'].dropna().values
            for d in district_medians.index]
    bp = ax.boxplot(data, labels=district_medians.index, patch_artist=True)

    # Color boxes - prime areas in coral
    prime = ['SW1', 'SW3', 'SW7', 'W1', 'W8', 'NW1', 'NW3', 'NW8']
    colors = ['lightcoral' if any(d.startswith(p) for p in prime) else 'lightblue'
              for d in district_medians.index]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Subject marker
    subject_ppsf = subject['predicted_pcm'] / subject['size_sqft']
    if subject_district in district_medians.index:
        idx = list(district_medians.index).index(subject_district) + 1
        ax.scatter([idx], [subject_ppsf], s=200, c='red', marker='*',
                   zorder=10, label='Subject Property')

    ax.set_xlabel('District', fontsize=12)
    ax.set_ylabel('Price per Sqft (£/sqft)', fontsize=12)
    ax.set_title('Rent by District (£/sqft) - Prime Areas in Red', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def chart_closest_comps(comps, subject, output_path, top_n=20):
    """Chart 4: Horizontal bar chart of top-scoring comparables by tier."""
    fig, ax = plt.subplots(figsize=(12, 8))

    comps = comps.copy()

    # Sort by composite score (location 50%, size 25%, PPSF 25%) if available
    if 'comp_score' in comps.columns:
        closest = comps.nlargest(top_n, 'comp_score')
        chart_title = f'Top {top_n} Best-Matched Comparables (Location + Size + £/sqft)'
    else:
        # Fallback to old method
        comps['size_diff'] = abs(comps['size_sqft'] - subject['size_sqft'])
        closest = comps.nsmallest(top_n, 'size_diff')
        chart_title = f'Top {top_n} Closest Comparables by Size'

    # Create labels with tier indicator if available
    labels = []
    for _, row in closest.iterrows():
        addr = f"{row['address'][:35]}..." if len(row['address']) > 35 else row['address']
        if 'tier' in row and pd.notna(row['tier']):
            tier_label = row['tier'].replace('Tier ', 'T')  # "T1: Same District"
            labels.append(f"[{tier_label[:2]}] {addr}")
        else:
            labels.append(addr)

    # Colors based on tier (if available) or PPSF relative to subject
    tier_colors = {1: '#2E7D32', 2: '#1976D2', 3: '#F57C00', 4: '#757575'}  # Green, Blue, Orange, Gray
    subject_ppsf = subject['predicted_pcm'] / subject['size_sqft']

    if 'tier_num' in closest.columns:
        colors = [tier_colors.get(t, '#757575') for t in closest['tier_num']]
    else:
        colors = ['lightcoral' if p > subject_ppsf else 'lightgreen' for p in closest['ppsf']]

    y_pos = np.arange(len(closest))
    bars = ax.barh(y_pos, closest['price_pcm'], color=colors, edgecolor='white', alpha=0.8)

    # Add PPSF labels with score if available
    for i, (_, row) in enumerate(closest.iterrows()):
        if 'comp_score' in row and pd.notna(row['comp_score']):
            ax.text(row['price_pcm'] + 100, i, f"£{row['ppsf']:.2f}/sqft (score: {row['comp_score']:.0f})",
                    va='center', fontsize=8)
        else:
            ax.text(row['price_pcm'] + 100, i, f"£{row['ppsf']:.2f}/sqft",
                    va='center', fontsize=8)

    # Subject line
    ax.axvline(subject['predicted_pcm'], color='red', linestyle='--', linewidth=2,
               label=f"Model Valuation: £{subject['predicted_pcm']:,}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Monthly Rent (£)', fontsize=12)
    ax.set_title(chart_title, fontsize=14, fontweight='bold')

    # Add legend for tiers if using tiered system
    if 'tier_num' in closest.columns:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E7D32', label='T1: Same District'),
            Patch(facecolor='#1976D2', label='T2: Same Area (SW1)'),
            Patch(facecolor='#F57C00', label='T3: Prime Adjacent'),
            Patch(facecolor='#757575', label='T4: Broader Market'),
            plt.Line2D([0], [0], color='red', linestyle='--', label=f"Valuation: £{subject['predicted_pcm']:,}")
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    else:
        ax.legend(loc='lower right')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('£{x:,.0f}'))
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_html_report(subject, comps, stats, output_path):
    """Generate the HTML report with embedded charts."""

    subject_ppsf = subject['predicted_pcm'] / subject['size_sqft']

    # Build tiered comparables tables
    def build_comp_rows(df, max_rows=5):
        rows = ''
        for _, row in df.head(max_rows).iterrows():
            ppsf_diff = ((row['ppsf'] / subject_ppsf) - 1) * 100
            if ppsf_diff > 10:
                color = '#fed7d7'
            elif ppsf_diff < -10:
                color = '#c6f6d5'
            else:
                color = 'white'
            rows += f'''
            <tr style="background:{color}">
                <td><a href="{row['url']}" target="_blank">{str(row['address'])[:45]}</a></td>
                <td>{row['district']}</td>
                <td>£{row['price_pcm']:,.0f}</td>
                <td>{row['size_sqft']:,.0f}</td>
                <td>£{row['ppsf']:.2f}</td>
                <td>{row['source']}</td>
            </tr>
            '''
        return rows

    # Build tier sections
    tier_sections = ''
    tier_info = [
        (1, 'Tier 1: Same District (SW1W)', 'Direct comparables in the same postcode district', '#e6fffa'),
        (2, 'Tier 2: SW1 Area', 'Other SW1 districts (SW1X, SW1V, SW1A, etc.) - similar prime Belgravia/Westminster market', '#ebf8ff'),
        (3, 'Tier 3: Prime Central London', 'Adjacent prime areas (SW3 Chelsea, SW7 South Ken, W1 Mayfair, W8 Kensington)', '#faf5ff'),
        (4, 'Tier 4: Broader Market', 'Properties of similar size across wider London market', '#f7fafc'),
    ]

    for tier_num, tier_name, tier_desc, tier_bg in tier_info:
        if 'tier_num' in comps.columns:
            tier_comps = comps[comps['tier_num'] == tier_num]
        else:
            tier_comps = pd.DataFrame()

        if len(tier_comps) > 0:
            tier_median_ppsf = stats.get(f'tier{tier_num}_median_ppsf', 'N/A')
            tier_median_rent = stats.get(f'tier{tier_num}_median_rent', 'N/A')
            tier_sections += f'''
            <div style="background:{tier_bg};padding:15px;border-radius:8px;margin-bottom:15px;">
                <h4 style="margin:0 0 5px 0;color:#1a365d;">{tier_name} ({len(tier_comps)} properties)</h4>
                <p style="margin:0 0 10px 0;font-size:12px;color:#4a5568;">{tier_desc}</p>
                <p style="margin:0 0 10px 0;font-size:13px;"><strong>Tier Median: £{tier_median_ppsf}/sqft | £{tier_median_rent:,.0f}/month</strong></p>
                <table style="margin:0;">
                    <thead>
                        <tr>
                            <th>Address</th>
                            <th>District</th>
                            <th>Rent</th>
                            <th>Size</th>
                            <th>£/sqft</th>
                            <th>Source</th>
                        </tr>
                    </thead>
                    <tbody>
                        {build_comp_rows(tier_comps, 5)}
                    </tbody>
                </table>
                {f'<p style="font-size:11px;color:#718096;margin-top:5px;">Showing 5 of {len(tier_comps)} | Full list in comparables.csv</p>' if len(tier_comps) > 5 else ''}
            </div>
            '''

    # Fallback if no tier data
    if not tier_sections:
        tier_sections = '<p>No tiered comparable data available.</p>'

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Property Valuation - {subject['address']}</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2d5a87 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 25px;
            text-align: center;
        }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
        .header p {{ margin: 5px 0; opacity: 0.9; }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .card h2 {{
            margin-top: 0;
            color: #1a365d;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }}
        .valuation-box {{
            background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
            border: 2px solid #3182ce;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .valuation-box .amount {{
            font-size: 48px;
            font-weight: bold;
            color: #2b6cb0;
        }}
        .valuation-box .range {{
            font-size: 16px;
            color: #4a5568;
            margin-top: 10px;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 24px;
            font-weight: bold;
            color: #2d5a87;
        }}
        .stat-box .label {{
            font-size: 11px;
            color: #718096;
            text-transform: uppercase;
            margin-top: 5px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}
        th, td {{
            padding: 12px 10px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f7fafc;
            font-weight: 600;
            color: #4a5568;
        }}
        tr:hover {{ background: #f7fafc; }}
        a {{ color: #3182ce; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #718096;
            font-size: 12px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .methodology {{
            background: #f7fafc;
            border-radius: 8px;
            padding: 15px;
            font-size: 13px;
            color: #4a5568;
        }}
        @media (max-width: 900px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Property Valuation Report</h1>
        <p><strong>{subject['address']}, London {subject['postcode']}</strong></p>
        <p style="margin-top:15px; font-size:14px; opacity:0.85">
            Generated: {datetime.now().strftime('%d %B %Y')} | Model: XGBoost V14 (46 features) | Market Data: {stats['total_market']} listings
        </p>
    </div>

    <div class="valuation-box">
        <div style="font-size:14px;color:#4a5568;margin-bottom:5px;">ESTIMATED MONTHLY RENT</div>
        <div class="amount">£{subject['predicted_pcm']:,}</div>
        <div class="range">
            Confidence Range: £{subject['range_low']:,} - £{subject['range_high']:,} pcm
        </div>
        <div style="margin-top:10px;font-size:13px;color:#718096;">
            £{subject_ppsf:.2f}/sqft | {stats['market_percentile']:.0f}th percentile of market
        </div>
    </div>

    <div class="card">
        <h2>Property Details</h2>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="value">{subject['size_sqft']:,}</div>
                <div class="label">Square Feet</div>
            </div>
            <div class="stat-box">
                <div class="value">{subject['bedrooms']}</div>
                <div class="label">Bedrooms</div>
            </div>
            <div class="stat-box">
                <div class="value">{subject['bathrooms']}</div>
                <div class="label">Bathrooms</div>
            </div>
            <div class="stat-box">
                <div class="value">{subject['postcode']}</div>
                <div class="label">Postcode</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Market Position</h2>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="value">£{subject_ppsf:.2f}</div>
                <div class="label">Subject £/sqft</div>
            </div>
            <div class="stat-box">
                <div class="value">£{stats['median_ppsf']}</div>
                <div class="label">Comp Median £/sqft</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['market_percentile']:.0f}th</div>
                <div class="label">Market Percentile</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['total_comps']}</div>
                <div class="label">Comparables</div>
            </div>
        </div>
        <p style="font-size:13px;color:#718096;text-align:center;margin-top:15px;">
            The property's £/sqft places it in the <strong>{stats['market_percentile']:.0f}th percentile</strong> of the London rental market.
            {'This is a premium valuation typical for Belgravia/SW1.' if stats['market_percentile'] > 70 else ''}
        </p>
    </div>

    <div class="card">
        <h2>Market Analysis</h2>
        <div class="charts-grid">
            <div class="chart-container">
                <img src="chart1_ppsf_distribution.png" alt="PPSF Distribution">
            </div>
            <div class="chart-container">
                <img src="chart2_size_vs_price.png" alt="Size vs Price">
            </div>
        </div>
        <div class="charts-grid">
            <div class="chart-container">
                <img src="chart3_price_by_district.png" alt="Price by District">
            </div>
            <div class="chart-container">
                <img src="chart4_closest_comps.png" alt="Closest Comparables">
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Comparable Properties by Location Tier</h2>
        <p style="font-size:13px;color:#718096;margin-bottom:15px;">
            Properties similar in size (±20%) with {subject['bedrooms']} bedrooms, ranked by location relevance.
            <span style="background:#c6f6d5;padding:2px 8px;border-radius:4px;">Green</span> = lower £/sqft,
            <span style="background:#fed7d7;padding:2px 8px;border-radius:4px;">Red</span> = higher £/sqft
        </p>
        {tier_sections}
        <p style="font-size:12px;color:#718096;margin-top:15px;">Total: {stats['total_comps']} comparables across all tiers. Full list in comparables.csv</p>
    </div>

    <div class="card">
        <h2>Market Statistics</h2>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="value">£{stats['p25_ppsf']}</div>
                <div class="label">25th Percentile £/sqft</div>
            </div>
            <div class="stat-box">
                <div class="value">£{stats['median_ppsf']}</div>
                <div class="label">Median £/sqft</div>
            </div>
            <div class="stat-box">
                <div class="value">£{stats['p75_ppsf']}</div>
                <div class="label">75th Percentile £/sqft</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats['sw1_comps']}</div>
                <div class="label">SW1 Comparables</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Methodology</h2>
        <div class="methodology">
            <p><strong>Model:</strong> XGBoost regression with 46 features including size, location encodings, amenities, and property type indicators.</p>
            <p><strong>Training Data:</strong> {stats['total_market']} active London rental listings from multiple agents (Savills, Knight Frank, Foxtons, Chestertons, Rightmove).</p>
            <p><strong>Features:</strong> Size, bedrooms, bathrooms, postcode encoding, prime district indicator, amenities (AC, wood floors, balcony, etc.), property type (period, penthouse), and location scores.</p>
            <p><strong>Confidence Range:</strong> Based on model's ~18% median absolute percentage error.</p>
            <p><strong>Comparables:</strong> Properties within ±20% of subject size, matching bedroom count, from active listings.</p>
        </div>
    </div>

    <div class="footer">
        <p>Property Valuation Model V14 | Generated {datetime.now().strftime('%d %B %Y')}</p>
        <p>{subject['address']}, {subject['postcode']}</p>
        <p style="margin-top:10px;font-size:11px;color:#a0aec0;">
            Disclaimer: This is an automated valuation estimate (AVM). Actual rental values depend on property condition,
            exact fixtures/fittings, lease terms, and current market conditions. This should be used as guidance only.
        </p>
    </div>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Generate property valuation report')
    parser.add_argument('--address', default=SUBJECT['address'])
    parser.add_argument('--postcode', default=SUBJECT['postcode'])
    parser.add_argument('--size', type=int, default=SUBJECT['size_sqft'])
    parser.add_argument('--beds', type=int, default=SUBJECT['bedrooms'])
    parser.add_argument('--baths', type=int, default=SUBJECT['bathrooms'])
    parser.add_argument('--predicted', type=int, default=SUBJECT['predicted_pcm'])
    parser.add_argument('--low', type=int, default=SUBJECT['range_low'])
    parser.add_argument('--high', type=int, default=SUBJECT['range_high'])

    args = parser.parse_args()

    subject = {
        'address': args.address,
        'postcode': args.postcode,
        'size_sqft': args.size,
        'bedrooms': args.beds,
        'bathrooms': args.baths,
        'predicted_pcm': args.predicted,
        'range_low': args.low,
        'range_high': args.high,
    }

    print(f"\n{'='*60}")
    print("GENERATING PROPERTY VALUATION REPORT")
    print(f"{'='*60}")
    print(f"Property: {subject['address']}, {subject['postcode']}")
    print(f"Size: {subject['size_sqft']} sqft | {subject['bedrooms']} bed | {subject['bathrooms']} bath")
    print(f"Model Valuation: £{subject['predicted_pcm']:,}/month")
    print(f"Range: £{subject['range_low']:,} - £{subject['range_high']:,}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading comparables from database...")
    comps = load_comparables(subject)
    print(f"  Found {len(comps)} comparable properties")

    print("Loading all listings for distribution analysis...")
    all_listings = load_all_listings_with_ppsf()
    print(f"  Found {len(all_listings)} total listings")

    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calc_statistics(comps, all_listings, subject)
    print(f"  Subject PPSF: £{stats['subject_ppsf']}/sqft")
    print(f"  Market Percentile: {stats['market_percentile']:.0f}th")

    # Generate charts
    print("\nGenerating charts...")
    chart_ppsf_distribution(all_listings, subject, OUTPUT_DIR / 'chart1_ppsf_distribution.png')
    chart_size_vs_price(comps, subject, OUTPUT_DIR / 'chart2_size_vs_price.png')
    chart_price_by_district(all_listings, subject, OUTPUT_DIR / 'chart3_price_by_district.png')
    chart_closest_comps(comps, subject, OUTPUT_DIR / 'chart4_closest_comps.png')

    # Save comparables CSV
    print("\nSaving data files...")
    comps.to_csv(OUTPUT_DIR / 'comparables.csv', index=False)
    print(f"  Saved: comparables.csv")

    # Save statistics JSON
    with open(OUTPUT_DIR / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: statistics.json")

    # Generate HTML report
    print("\nGenerating HTML report...")
    generate_html_report(subject, comps, stats, OUTPUT_DIR / 'negotiation_report.html')

    print(f"\n{'='*60}")
    print("REPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Open: {OUTPUT_DIR / 'negotiation_report.html'}")
    print()


if __name__ == '__main__':
    main()
