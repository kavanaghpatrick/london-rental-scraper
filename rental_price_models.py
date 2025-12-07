"""
London Rental Price Prediction Models

Built using recommendations from Grok and Gemini analysis.
Models: Ridge, Random Forest, XGBoost, Ensemble

Usage:
    python rental_price_models.py
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fall back to GradientBoosting if not available
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using GradientBoostingRegressor instead")

# Try to import category_encoders for target encoding
try:
    from category_encoders import TargetEncoder
    HAS_TARGET_ENCODER = True
except ImportError:
    HAS_TARGET_ENCODER = False
    print("category_encoders not available, using fallback encoding")


class TargetEncoderFallback(BaseEstimator, TransformerMixin):
    """Simple target encoder fallback if category_encoders not installed."""

    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.encodings = {}
        self.global_mean = None

    def fit(self, X, y):
        self.global_mean = np.mean(y)
        X = pd.DataFrame(X)
        for col in X.columns:
            stats = pd.DataFrame({'feature': X[col], 'target': y}).groupby('feature')['target'].agg(['mean', 'count'])
            smoothed = (stats['mean'] * stats['count'] + self.global_mean * self.smoothing) / (stats['count'] + self.smoothing)
            self.encodings[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        result = X.copy()
        for col in X.columns:
            result[col] = X[col].map(self.encodings.get(col, {})).fillna(self.global_mean)
        return result.values


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Avoid division by zero for bedrooms
        beds_adj = X['bedrooms'].replace(0, 1)

        # Avg room size
        X['avg_room_size'] = X['size_sqft'] / beds_adj

        # Bath ratio (luxury proxy)
        X['bath_ratio'] = X['bathrooms'].fillna(1) / beds_adj

        # Log size (captures diminishing returns)
        X['log_sqft'] = np.log1p(X['size_sqft'])

        return X


def load_data():
    """Load and prepare data from SQLite database."""
    conn = sqlite3.connect('output/rentals.db')

    query = """
    SELECT
        bedrooms,
        bathrooms,
        size_sqft,
        postcode,
        area,
        property_type,
        price_pcm
    FROM listings_deduped
    WHERE size_sqft > 0
        AND bedrooms IS NOT NULL
        AND price_pcm > 0
        AND price_pcm < 100000
        AND size_sqft < 15000
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Clean area names (normalize case)
    df['area'] = df['area'].str.title().str.strip()
    df['area'] = df['area'].replace('', 'Unknown')
    df['area'] = df['area'].fillna('Unknown')

    # Clean postcode (take first part)
    df['postcode'] = df['postcode'].fillna('Unknown')
    df['postcode_district'] = df['postcode'].str.extract(r'^([A-Z]{1,2}\d{1,2})', expand=False)
    df['postcode_district'] = df['postcode_district'].fillna('Other')

    # Clean property type
    df['property_type'] = df['property_type'].fillna('unknown')
    df['property_type'] = df['property_type'].str.lower()

    print(f"Loaded {len(df)} records")
    print(f"Price range: £{df['price_pcm'].min():,.0f} - £{df['price_pcm'].max():,.0f}")
    print(f"Size range: {df['size_sqft'].min():.0f} - {df['size_sqft'].max():.0f} sqft")

    return df


def create_preprocessor():
    """Create the preprocessing pipeline."""

    num_features = ['bedrooms', 'bathrooms', 'size_sqft', 'avg_room_size', 'bath_ratio', 'log_sqft']
    cat_features_high = ['postcode_district', 'area']
    cat_features_low = ['property_type']

    # Choose encoder based on availability
    if HAS_TARGET_ENCODER:
        high_card_encoder = TargetEncoder(smoothing=10)
    else:
        high_card_encoder = TargetEncoderFallback(smoothing=10)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_features),

            ('cat_high', high_card_encoder, cat_features_high),

            ('cat_low', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_features_low)
        ],
        remainder='drop'
    )

    return preprocessor


def build_models():
    """Build and return all model pipelines."""

    models = {}

    # Feature engineering + preprocessing pipeline
    prep_pipeline = Pipeline([
        ('engineer', FeatureEngineer()),
        ('preprocessor', create_preprocessor())
    ])

    # Model 1: Ridge Regression (baseline)
    models['Ridge'] = {
        'pipeline': Pipeline([
            ('prep', prep_pipeline),
            ('model', Ridge())
        ]),
        'params': {'model__alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
    }

    # Model 2: Lasso Regression (feature selection)
    models['Lasso'] = {
        'pipeline': Pipeline([
            ('prep', prep_pipeline),
            ('model', Lasso(max_iter=5000))
        ]),
        'params': {'model__alpha': [0.1, 1.0, 10.0, 50.0]}
    }

    # Model 3: Random Forest
    models['RandomForest'] = {
        'pipeline': Pipeline([
            ('prep', prep_pipeline),
            ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
        ]),
        'params': {
            'model__n_estimators': [200, 500],
            'model__max_depth': [10, 20],
            'model__min_samples_leaf': [2, 5]
        }
    }

    # Model 4: XGBoost or GradientBoosting
    if HAS_XGBOOST:
        models['XGBoost'] = {
            'pipeline': Pipeline([
                ('prep', prep_pipeline),
                ('model', XGBRegressor(random_state=42, n_jobs=-1, verbosity=0))
            ]),
            'params': {
                'model__n_estimators': [300, 500],
                'model__learning_rate': [0.01, 0.05],
                'model__max_depth': [3, 5],
                'model__subsample': [0.7, 0.8]
            }
        }
    else:
        models['GradientBoosting'] = {
            'pipeline': Pipeline([
                ('prep', prep_pipeline),
                ('model', GradientBoostingRegressor(random_state=42))
            ]),
            'params': {
                'model__n_estimators': [200, 500],
                'model__learning_rate': [0.01, 0.05],
                'model__max_depth': [3, 5]
            }
        }

    return models


def evaluate_model(model, X, y, cv=5):
    """Evaluate model using cross-validation on log-transformed target."""

    # Log-transform target
    y_log = np.log1p(y)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    scores = {
        'mae': [],
        'rmse': [],
        'r2': [],
        'mape': []
    }

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]
        y_val_actual = y.iloc[val_idx]

        # Fit on log-transformed target
        model.fit(X_train, y_train_log)

        # Predict and inverse transform
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)

        # Calculate metrics
        scores['mae'].append(mean_absolute_error(y_val_actual, y_pred))
        scores['rmse'].append(np.sqrt(mean_squared_error(y_val_actual, y_pred)))
        scores['r2'].append(r2_score(y_val_actual, y_pred))
        scores['mape'].append(np.mean(np.abs((y_val_actual - y_pred) / y_val_actual)) * 100)

    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}


def simple_grid_search(model, param_grid, X, y, cv=3):
    """Simple grid search for best hyperparameters."""
    from itertools import product

    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_score = float('inf')
    best_params = None

    y_log = np.log1p(y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for combo in product(*values):
        params = dict(zip(keys, combo))
        model.set_params(**params)

        # Quick CV score
        mae_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_log.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = np.expm1(model.predict(X_val))
            mae_scores.append(mean_absolute_error(y_val, y_pred))

        mean_mae = np.mean(mae_scores)
        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params.copy()

    # Return model with best params
    model.set_params(**best_params)
    return model, best_params, best_score


def main():
    print("=" * 70)
    print("LONDON RENTAL PRICE PREDICTION MODELS")
    print("=" * 70)

    # Load data
    df = load_data()

    # Prepare features and target
    X = df.drop('price_pcm', axis=1)
    y = df['price_pcm']

    print(f"\nFeatures: {list(X.columns)}")
    print(f"Target: price_pcm")
    print(f"Samples: {len(X)}")

    # Build models
    models = build_models()

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 70)

    results = {}
    trained_models = {}

    for name, config in models.items():
        print(f"\n[{name}] Tuning hyperparameters...")

        # Simple grid search
        best_model, best_params, best_cv_mae = simple_grid_search(
            config['pipeline'],
            config['params'],
            X, y,
            cv=3
        )

        print(f"  Best params: {best_params}")
        print(f"  CV MAE: £{best_cv_mae:,.0f}")

        # Full evaluation with best model
        print(f"  Running full 5-fold CV...")
        scores = evaluate_model(best_model, X, y, cv=5)

        results[name] = {
            'MAE': scores['mae'],
            'RMSE': scores['rmse'],
            'R2': scores['r2'],
            'MAPE': scores['mape'],
            'best_params': best_params
        }

        # Train final model on all data
        best_model.fit(X, np.log1p(y))
        trained_models[name] = best_model

        print(f"  MAE:  £{scores['mae'][0]:,.0f} (+/- £{scores['mae'][1]:,.0f})")
        print(f"  RMSE: £{scores['rmse'][0]:,.0f} (+/- £{scores['rmse'][1]:,.0f})")
        print(f"  R2:   {scores['r2'][0]:.3f} (+/- {scores['r2'][1]:.3f})")
        print(f"  MAPE: {scores['mape'][0]:.1f}% (+/- {scores['mape'][1]:.1f}%)")

    # Create ensemble
    print("\n" + "-" * 70)
    print("[Ensemble] Building weighted voting regressor...")

    # Get the best performing models for ensemble
    sorted_models = sorted(results.items(), key=lambda x: x[1]['MAE'][0])

    # Use top 3 models for ensemble with weights inversely proportional to MAE
    top_models = sorted_models[:3]
    total_inv_mae = sum(1/r[1]['MAE'][0] for r in top_models)

    estimators = []
    weights = []
    for name, metrics in top_models:
        weight = (1/metrics['MAE'][0]) / total_inv_mae
        estimators.append((name.lower(), trained_models[name]))
        weights.append(weight)
        print(f"  {name}: weight={weight:.2f}")

    # Create ensemble
    ensemble = VotingRegressor(estimators=estimators, weights=weights)

    # Evaluate ensemble
    print("  Running 5-fold CV on ensemble...")

    # Need to recreate models for ensemble evaluation
    ensemble_scores = {
        'mae': [],
        'rmse': [],
        'r2': [],
        'mape': []
    }

    y_log = np.log1p(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_log = y_log.iloc[train_idx]
        y_val_actual = y.iloc[val_idx]

        # Fit ensemble
        ensemble.fit(X_train, y_train_log)

        # Predict
        y_pred = np.expm1(ensemble.predict(X_val))

        ensemble_scores['mae'].append(mean_absolute_error(y_val_actual, y_pred))
        ensemble_scores['rmse'].append(np.sqrt(mean_squared_error(y_val_actual, y_pred)))
        ensemble_scores['r2'].append(r2_score(y_val_actual, y_pred))
        ensemble_scores['mape'].append(np.mean(np.abs((y_val_actual - y_pred) / y_val_actual)) * 100)

    results['Ensemble'] = {
        'MAE': (np.mean(ensemble_scores['mae']), np.std(ensemble_scores['mae'])),
        'RMSE': (np.mean(ensemble_scores['rmse']), np.std(ensemble_scores['rmse'])),
        'R2': (np.mean(ensemble_scores['r2']), np.std(ensemble_scores['r2'])),
        'MAPE': (np.mean(ensemble_scores['mape']), np.std(ensemble_scores['mape']))
    }

    print(f"  MAE:  £{results['Ensemble']['MAE'][0]:,.0f} (+/- £{results['Ensemble']['MAE'][1]:,.0f})")
    print(f"  RMSE: £{results['Ensemble']['RMSE'][0]:,.0f}")
    print(f"  R2:   {results['Ensemble']['R2'][0]:.3f}")
    print(f"  MAPE: {results['Ensemble']['MAPE'][0]:.1f}%")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'R2':>10} {'MAPE':>10}")
    print("-" * 70)

    for name in sorted(results.keys(), key=lambda x: results[x]['MAE'][0]):
        r = results[name]
        print(f"{name:<20} £{r['MAE'][0]:>10,.0f} £{r['RMSE'][0]:>10,.0f} {r['R2'][0]:>10.3f} {r['MAPE'][0]:>9.1f}%")

    # Best model
    best = min(results.items(), key=lambda x: x[1]['MAE'][0])
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best[0]}")
    print(f"  Mean Absolute Error: £{best[1]['MAE'][0]:,.0f}/month")
    print(f"  On avg £8,091 rent, that's {best[1]['MAPE'][0]:.1f}% error")
    print("=" * 70)

    # Example predictions
    print("\n[EXAMPLE PREDICTIONS]")
    sample_properties = [
        {'bedrooms': 2, 'bathrooms': 1, 'size_sqft': 800, 'postcode': 'SW3', 'area': 'Chelsea', 'property_type': 'flat'},
        {'bedrooms': 3, 'bathrooms': 2, 'size_sqft': 1500, 'postcode': 'W8', 'area': 'Kensington', 'property_type': 'flat'},
        {'bedrooms': 1, 'bathrooms': 1, 'size_sqft': 500, 'postcode': 'NW3', 'area': 'Hampstead', 'property_type': 'flat'},
    ]

    # Train ensemble on full data
    ensemble.fit(X, np.log1p(y))

    for prop in sample_properties:
        prop_df = pd.DataFrame([prop])
        prop_df['postcode_district'] = prop['postcode']
        pred = np.expm1(ensemble.predict(prop_df))[0]
        print(f"  {prop['bedrooms']}bed/{prop['size_sqft']}sqft in {prop['area']}: £{pred:,.0f}/month")

    return results, trained_models, ensemble


if __name__ == '__main__':
    results, models, ensemble = main()
