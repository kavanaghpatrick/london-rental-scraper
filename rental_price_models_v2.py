"""
London Rental Price Prediction Models V2 - Improved

Improvements based on Grok and Gemini analysis:
1. Data cleaning: Remove outliers and impossible entries
2. Feature engineering: Interactions, polynomial terms
3. Model: Deeper trees, MAE loss for robustness
4. Quantile regression for prediction intervals

Usage:
    python rental_price_models_v2.py
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data():
    """Load data and apply cleaning rules from Gemini analysis."""
    conn = sqlite3.connect('output/rentals.db')
    df = pd.read_sql("""
        SELECT bedrooms, bathrooms, size_sqft, postcode, area, property_type, price_pcm
        FROM listings_deduped
        WHERE size_sqft > 0 AND bedrooms IS NOT NULL AND price_pcm > 0
    """, conn)
    conn.close()

    print(f"Raw data: {len(df)} records")

    # CLEANING RULE 1: Remove impossible sqft/bedroom ratios
    # < 70 sqft per bedroom is likely data error
    sqft_per_bed = df['size_sqft'] / df['bedrooms'].replace(0, 0.5)
    mask_density = sqft_per_bed < 70
    print(f"  Removed {mask_density.sum()} with <70 sqft/bedroom")

    # CLEANING RULE 2: Remove price/sqft outliers
    ppsqft = df['price_pcm'] / df['size_sqft']
    mask_ppsqft = (ppsqft < 1.0) | (ppsqft > 25.0)
    print(f"  Removed {mask_ppsqft.sum()} with price/sqft outliers")

    # CLEANING RULE 3: Cap extreme prices (>£50k likely errors or penthouses)
    mask_cap = df['price_pcm'] > 50000
    print(f"  Removed {mask_cap.sum()} with price >£50k")

    # CLEANING RULE 4: Remove extremely small properties (<150 sqft)
    mask_small = df['size_sqft'] < 150
    print(f"  Removed {mask_small.sum()} with size <150 sqft")

    # Apply all filters
    df_clean = df[~(mask_density | mask_ppsqft | mask_cap | mask_small)].copy()
    print(f"Clean data: {len(df_clean)} records ({100*len(df_clean)/len(df):.1f}% retained)")

    return df_clean


def engineer_features(df):
    """Apply feature engineering from Grok and Gemini analysis."""

    # Basic cleaning
    df['area'] = df['area'].str.title().fillna('Unknown')
    df['postcode_district'] = df['postcode'].str.extract(r'^([A-Z]{1,2}\d{1,2})')[0].fillna('Other')
    df['property_type'] = df['property_type'].fillna('unknown').str.lower()
    df['bathrooms'] = df['bathrooms'].fillna(1)

    # FEATURE 1: Size per bedroom (from Grok)
    beds_adj = df['bedrooms'].replace(0, 0.5)
    df['size_per_bed'] = df['size_sqft'] / beds_adj

    # FEATURE 2: Bath ratio (luxury proxy)
    df['bath_ratio'] = df['bathrooms'] / beds_adj

    # FEATURE 3: Bed-bath interaction (from Gemini)
    df['bed_bath_interaction'] = df['bedrooms'] * df['bathrooms']

    # FEATURE 4: Polynomial size (from Gemini - captures non-linear size premium)
    df['size_poly'] = df['size_sqft'] ** 1.5 / 1000  # Scale down

    # FEATURE 5: Sqft per bathroom (spaciousness proxy from Gemini)
    df['sqft_per_bath'] = df['size_sqft'] / (df['bathrooms'] + 1)

    # FEATURE 6: Log size (diminishing returns)
    df['log_sqft'] = np.log1p(df['size_sqft'])

    # FEATURE 7: Size-postcode interaction (from Grok)
    # Will be created after target encoding

    # Target encoding for location features
    global_mean = np.log1p(df['price_pcm']).mean()
    smoothing = 15  # Higher smoothing for robustness

    for col in ['area', 'postcode_district']:
        means = np.log1p(df.groupby(col)['price_pcm'].transform('mean'))
        counts = df.groupby(col)['price_pcm'].transform('count')
        df[f'{col}_encoded'] = (means * counts + global_mean * smoothing) / (counts + smoothing)

    # Size-postcode interaction
    df['size_postcode_interaction'] = df['size_sqft'] * df['postcode_district_encoded'] / 1000

    # One-hot encode property type
    df = pd.get_dummies(df, columns=['property_type'], prefix='type')

    return df


def build_improved_xgboost():
    """Build XGBoost with improved hyperparameters from Grok/Gemini."""
    return XGBRegressor(
        n_estimators=800,
        learning_rate=0.015,
        max_depth=6,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='reg:absoluteerror',  # MAE for robustness (Gemini suggestion)
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )


def build_quantile_model(quantile):
    """Build quantile regression model for prediction intervals."""
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.7,
        objective='reg:quantileerror',
        quantile_alpha=quantile,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )


def evaluate_model(model, X, y, cv=5):
    """Evaluate model with detailed metrics."""
    y_log = np.log1p(y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    all_preds = []
    all_actual = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = np.expm1(model.predict(X_val))

        all_preds.extend(y_pred)
        all_actual.extend(y_val.values)

    all_preds = np.array(all_preds)
    all_actual = np.array(all_actual)

    mae = mean_absolute_error(all_actual, all_preds)
    rmse = np.sqrt(mean_squared_error(all_actual, all_preds))
    r2 = r2_score(all_actual, all_preds)
    mape = np.mean(np.abs((all_actual - all_preds) / all_actual)) * 100
    median_ape = np.median(np.abs((all_actual - all_preds) / all_actual)) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Median_APE': median_ape,
        'predictions': all_preds,
        'actual': all_actual
    }


def analyze_errors_by_tier(actual, preds, tiers):
    """Analyze prediction errors by price tier."""
    print("\n  Error by Price Tier:")
    for low, high, label in tiers:
        mask = (actual >= low) & (actual < high)
        if mask.sum() > 0:
            tier_preds = preds[mask]
            tier_actual = actual[mask]
            tier_mape = np.mean(np.abs((tier_actual - tier_preds) / tier_actual)) * 100
            tier_mae = np.mean(np.abs(tier_actual - tier_preds))
            print(f"    {label}: n={mask.sum()}, MAPE={tier_mape:.1f}%, MAE=£{tier_mae:,.0f}")


def main():
    print("=" * 70)
    print("LONDON RENTAL PRICE MODELS V2 - IMPROVED")
    print("Based on Grok and Gemini recommendations")
    print("=" * 70)

    # Load and clean data
    df = load_and_clean_data()

    # Engineer features
    print("\nEngineering features...")
    df = engineer_features(df)

    # Define feature columns
    feature_cols = [
        'bedrooms', 'bathrooms', 'size_sqft',
        'size_per_bed', 'bath_ratio', 'bed_bath_interaction',
        'size_poly', 'sqft_per_bath', 'log_sqft',
        'area_encoded', 'postcode_district_encoded',
        'size_postcode_interaction'
    ] + [c for c in df.columns if c.startswith('type_')]

    X = df[feature_cols].fillna(0)
    y = df['price_pcm']

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target: £{y.min():,.0f} - £{y.max():,.0f} (mean £{y.mean():,.0f})")

    # Define price tiers for analysis
    tiers = [
        (0, 3000, '<£3k'),
        (3000, 5000, '£3-5k'),
        (5000, 10000, '£5-10k'),
        (10000, 20000, '£10-20k'),
        (20000, 100000, '£20k+')
    ]

    # Build and evaluate models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (5-fold CV)")
    print("=" * 70)

    models = {
        'XGBoost_V2': build_improved_xgboost(),
        'RandomForest_Deep': RandomForestRegressor(
            n_estimators=500, max_depth=15, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.02, max_depth=5,
            min_samples_leaf=3, random_state=42
        )
    }

    results = {}
    for name, model in models.items():
        print(f"\n[{name}]")
        result = evaluate_model(model, X, y)
        results[name] = result

        print(f"  R²:         {result['R2']:.4f}")
        print(f"  MAE:        £{result['MAE']:,.0f}")
        print(f"  RMSE:       £{result['RMSE']:,.0f}")
        print(f"  MAPE:       {result['MAPE']:.1f}%")
        print(f"  Median APE: {result['Median_APE']:.1f}%")

        analyze_errors_by_tier(result['actual'], result['predictions'], tiers)

    # Best model analysis
    best_name = min(results, key=lambda x: results[x]['MAE'])
    best = results[best_name]

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}")
    print("=" * 70)
    print(f"R²:         {best['R2']:.4f}")
    print(f"MAE:        £{best['MAE']:,.0f}")
    print(f"MAPE:       {best['MAPE']:.1f}%")
    print(f"Median APE: {best['Median_APE']:.1f}%")

    # Train quantile models for prediction intervals
    print("\n" + "-" * 70)
    print("QUANTILE REGRESSION (Prediction Intervals)")
    print("-" * 70)

    y_log = np.log1p(y)

    quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        model = build_quantile_model(q)
        model.fit(X, y_log)
        quantile_models[q] = model
        print(f"  Trained q={q:.1f} model")

    # Example predictions with intervals
    print("\n[EXAMPLE PREDICTIONS WITH 80% INTERVALS]")
    sample_indices = np.random.choice(len(X), 5, replace=False)

    for idx in sample_indices:
        actual = y.iloc[idx]
        q10 = np.expm1(quantile_models[0.1].predict(X.iloc[[idx]]))[0]
        q50 = np.expm1(quantile_models[0.5].predict(X.iloc[[idx]]))[0]
        q90 = np.expm1(quantile_models[0.9].predict(X.iloc[[idx]]))[0]

        beds = df.iloc[idx]['bedrooms']
        sqft = df.iloc[idx]['size_sqft']
        area = df.iloc[idx]['area']

        print(f"  {int(beds)}bed/{int(sqft)}sqft in {area}:")
        print(f"    Actual: £{actual:,.0f}")
        print(f"    Predicted: £{q50:,.0f} [£{q10:,.0f} - £{q90:,.0f}]")

    # Feature importance
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (Best Model)")
    print("-" * 70)

    best_model = models[best_name]
    best_model.fit(X, np.log1p(y))

    if hasattr(best_model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        for _, row in importance.head(10).iterrows():
            bar = "█" * int(row['importance'] * 40)
            print(f"  {row['feature']:<30} {row['importance']:.3f} {bar}")

    # Improvement summary
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY vs V1")
    print("=" * 70)
    print("V1 (XGBoost baseline):  R²=0.658, MAE=£2,589, MAPE=28.2%")
    print(f"V2 ({best_name}):       R²={best['R2']:.3f}, MAE=£{best['MAE']:,.0f}, MAPE={best['MAPE']:.1f}%")

    r2_improvement = (best['R2'] - 0.658) / 0.658 * 100
    mae_improvement = (2589 - best['MAE']) / 2589 * 100
    mape_improvement = (28.2 - best['MAPE']) / 28.2 * 100

    print(f"\nImprovements:")
    print(f"  R² improved by:   {r2_improvement:+.1f}%")
    print(f"  MAE improved by:  {mae_improvement:+.1f}%")
    print(f"  MAPE improved by: {mape_improvement:+.1f}%")

    return results, quantile_models


if __name__ == '__main__':
    results, quantile_models = main()
