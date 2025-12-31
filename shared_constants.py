"""
Shared constants for data quality filters.

These thresholds are used consistently across:
- Model training (rental_price_models_v15.py)
- Dashboard queries (dashboard/src/lib/db.ts)
- Any future scripts

IMPORTANT: If you change these values, update db.ts manually
since it's TypeScript and can't import Python directly.
"""

# £/sqft thresholds for Prime Central London
PPSF_MIN = 3    # Below this = data error (impossible in PCL)
PPSF_MAX = 30   # Above this = extreme outlier or error

# Minimum rent to exclude storage/parking
PRICE_MIN_PCM = 500

# Size thresholds
SQFT_MIN = 150   # Studio minimum
SQFT_MAX = 10000 # Above this = likely error

# Sqft per bedroom sanity check
SQFT_PER_BED_MIN = 70  # Below this = impossible

# Prime Central London postcodes
PRIME_POSTCODES = ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW3', 'NW8']

# Premium agents (for feature encoding, not filtering)
PREMIUM_AGENTS = ['Knight Frank', 'Savills', 'Harrods Estates', 'Sotheby',
                  'Beauchamp Estates', 'Strutt & Parker', 'Chestertons']

# Source quality encoding (based on data completeness, NOT price)
SOURCE_QUALITY = {
    'savills': 4,      # Best sqft coverage
    'knightfrank': 4,
    'chestertons': 3,
    'foxtons': 2,
    'rightmove': 1     # Aggregator, often missing data
}
