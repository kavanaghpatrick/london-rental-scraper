"""
Rental Price Model Training Script (Postgres Version)

Trains the XGBoost model using data from Postgres, logs metrics to model_runs table,
and generates a report. Designed to run in GitHub Actions after daily scrapes.

Usage:
    python train_model_postgres.py
    python train_model_postgres.py --quick  # Skip Optuna tuning
"""

import os
import json
import time
import argparse
from datetime import datetime
import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not available - using default hyperparameters")


# ============ Configuration ============
TUBE_STATIONS = {
    'South Kensington': (51.4941, -0.1738),
    'Sloane Square': (51.4924, -0.1565),
    'Knightsbridge': (51.5015, -0.1607),
    'Hyde Park Corner': (51.5027, -0.1527),
    'Victoria': (51.4965, -0.1447),
}

PRIME_POSTCODES = ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW3', 'NW8']
PREMIUM_AGENTS = ['Knight Frank', 'Savills', 'Harrods Estates', 'Chestertons']
CITY_CENTER = (51.5074, -0.1278)


def get_connection():
    """Get Postgres connection."""
    url = os.environ.get('POSTGRES_URL')
    if not url:
        raise ValueError("POSTGRES_URL environment variable not set")
    return psycopg2.connect(url)


def load_data_from_postgres():
    """Load listings data from Postgres."""
    conn = get_connection()

    query = """
        SELECT
            id, source, address, postcode, bedrooms, bathrooms,
            size_sqft, price_pcm, latitude, longitude,
            features, description, agent_name, property_type,
            floor_count, has_ground, has_first_floor, has_second_floor,
            has_basement, has_roof_terrace
        FROM listings
        WHERE is_active = 1
          AND price_pcm > 500
          AND price_pcm < 100000
          AND size_sqft > 100
          AND size_sqft < 10000
          AND bedrooms > 0
          AND bedrooms <= 10
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Loaded {len(df)} listings from Postgres")
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df):
    """Engineer features for the model."""
    df = df.copy()

    # Postcode district
    df['postcode_district'] = df['postcode'].fillna('').str.extract(r'^([A-Z]+\d+)', expand=False)

    # Is prime location
    df['is_prime'] = df['postcode_district'].isin(PRIME_POSTCODES).astype(int)

    # Is premium agent
    df['is_premium_agent'] = df['agent_name'].fillna('').apply(
        lambda x: int(any(a.lower() in x.lower() for a in PREMIUM_AGENTS))
    )

    # Distance to center
    df['dist_to_center'] = df.apply(
        lambda r: haversine_distance(r['latitude'], r['longitude'], CITY_CENTER[0], CITY_CENTER[1])
        if pd.notna(r['latitude']) and r['latitude'] != 0 else np.nan,
        axis=1
    )

    # Nearest tube
    df['dist_to_tube'] = df.apply(
        lambda r: min(haversine_distance(r['latitude'], r['longitude'], lat, lon)
                     for lat, lon in TUBE_STATIONS.values())
        if pd.notna(r['latitude']) and r['latitude'] != 0 else np.nan,
        axis=1
    )

    # Size per bedroom
    df['sqft_per_bedroom'] = df['size_sqft'] / df['bedrooms'].clip(lower=1)

    # Amenity extraction from description
    desc = df['description'].fillna('').str.lower()
    df['has_balcony'] = desc.str.contains('balcony').astype(int)
    df['has_terrace'] = desc.str.contains('terrace').astype(int)
    df['has_garden'] = desc.str.contains('garden').astype(int)
    df['has_porter'] = (desc.str.contains('porter') | desc.str.contains('concierge')).astype(int)
    df['has_gym'] = desc.str.contains('gym').astype(int)
    df['has_parking'] = (desc.str.contains('parking') | desc.str.contains('garage')).astype(int)
    df['has_lift'] = (desc.str.contains('lift') | desc.str.contains('elevator')).astype(int)

    # One-hot encode postcode districts (top 15)
    top_districts = df['postcode_district'].value_counts().head(15).index
    for district in top_districts:
        df[f'pc_{district}'] = (df['postcode_district'] == district).astype(int)

    return df


def get_feature_columns(df):
    """Get list of feature columns."""
    base_features = [
        'bedrooms', 'bathrooms', 'size_sqft', 'sqft_per_bedroom',
        'is_prime', 'is_premium_agent',
        'dist_to_center', 'dist_to_tube',
        'has_balcony', 'has_terrace', 'has_garden', 'has_porter',
        'has_gym', 'has_parking', 'has_lift',
        'floor_count', 'has_ground', 'has_first_floor', 'has_second_floor',
        'has_basement', 'has_roof_terrace'
    ]

    # Add postcode one-hot columns
    pc_cols = [c for c in df.columns if c.startswith('pc_')]

    return base_features + pc_cols


def evaluate_model(model, X, y, n_splits=5):
    """Evaluate model with cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_preds = np.zeros(len(y))
    all_actual = y.values

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train on log-transformed target
        model.fit(X_train, np.log1p(y_train))
        preds = np.expm1(model.predict(X_val))
        all_preds[val_idx] = preds

    return {
        'R2': r2_score(all_actual, all_preds),
        'MAE': mean_absolute_error(all_actual, all_preds),
        'MAPE': np.mean(np.abs((all_actual - all_preds) / all_actual)) * 100,
        'Median_APE': np.median(np.abs((all_actual - all_preds) / all_actual)) * 100,
    }


def tune_optuna(X, y, n_trials=20):
    """Hyperparameter tuning with Optuna."""
    if not HAS_OPTUNA:
        return {}

    print(f"Running Optuna tuning ({n_trials} trials)...")
    y_log = np.log1p(y)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        }

        model = XGBRegressor(**params, objective='reg:absoluteerror',
                           random_state=42, n_jobs=-1, verbosity=0)

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model.fit(X.iloc[train_idx], y_log.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores.append(r2_score(y_log.iloc[val_idx], preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Best R² (log): {study.best_value:.4f}")
    return study.best_params


def log_model_run(metrics, samples, features_count, training_time, best_params=None):
    """Log model run to Postgres."""
    conn = get_connection()
    cur = conn.cursor()

    run_date = datetime.utcnow().strftime('%Y-%m-%d')
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    cur.execute('''
        INSERT INTO model_runs (
            run_date, run_id, version, samples_total, features_count,
            r2_score, mae, mape, median_ape, training_time_seconds, best_params
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        run_date, run_id, 'v15_postgres', samples, features_count,
        metrics['R2'], metrics['MAE'], metrics['MAPE'], metrics['Median_APE'],
        training_time, json.dumps(best_params) if best_params else None
    ))

    conn.commit()
    conn.close()
    print(f"Logged model run: {run_id}")
    return run_id


def get_model_history():
    """Get recent model performance history."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute('''
        SELECT run_date, r2_score, mae, mape, samples_total
        FROM model_runs
        ORDER BY created_at DESC
        LIMIT 10
    ''')

    rows = cur.fetchall()
    conn.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Skip Optuna tuning')
    args = parser.parse_args()

    print("=" * 60)
    print("RENTAL PRICE MODEL TRAINING (Postgres)")
    print("=" * 60)

    start_time = time.time()

    # Load and prepare data
    df = load_data_from_postgres()
    df = engineer_features(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0)
    y = df['price_pcm']

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")

    # Train model
    if args.quick or not HAS_OPTUNA:
        print("\nTraining with default hyperparameters...")
        model = XGBRegressor(
            n_estimators=1000, learning_rate=0.02, max_depth=6,
            subsample=0.8, colsample_bytree=0.7,
            objective='reg:absoluteerror', random_state=42, n_jobs=-1, verbosity=0
        )
        best_params = None
    else:
        best_params = tune_optuna(X, y, n_trials=20)
        model = XGBRegressor(
            **best_params, objective='reg:absoluteerror',
            random_state=42, n_jobs=-1, verbosity=0
        )

    # Evaluate
    metrics = evaluate_model(model, X, y)
    training_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"R²:         {metrics['R2']:.4f}")
    print(f"MAE:        £{metrics['MAE']:,.0f}")
    print(f"MAPE:       {metrics['MAPE']:.1f}%")
    print(f"Median APE: {metrics['Median_APE']:.1f}%")
    print(f"Time:       {training_time:.1f}s")

    # Log to database
    run_id = log_model_run(metrics, len(X), len(feature_cols), training_time, best_params)

    # Show history comparison
    print("\n" + "=" * 60)
    print("MODEL HISTORY (last 5 runs)")
    print("=" * 60)
    history = get_model_history()
    for row in history[:5]:
        print(f"  {row[0]}: R²={row[1]:.4f}, MAE=£{row[2]:,.0f}, MAPE={row[3]:.1f}%, n={row[4]}")

    # Check if model improved
    if len(history) > 1:
        current_r2 = metrics['R2']
        prev_r2 = history[1][1] if len(history) > 1 else 0
        change = current_r2 - prev_r2
        if change > 0.001:
            print(f"\n✅ Model IMPROVED by {change:.4f} R²")
        elif change < -0.001:
            print(f"\n⚠️ Model DEGRADED by {abs(change):.4f} R²")
        else:
            print(f"\n➡️ Model performance stable")

    return metrics


if __name__ == '__main__':
    main()
