"""
TASK #2 (model-eval): Segment error map for the v20 canonical model.

READ-ONLY analysis. NO retrain of the production artifact. We reuse the EXACT
retrain pipeline (retrain_canonical.load_recency_independent -> v20.engineer_features_v20
-> v20.get_feature_columns_v20) and produce OUT-OF-FOLD predictions via KFold
(n_splits=5, shuffle=True, random_state=42 -- matching retrain_canonical.cv_metrics)
so every row is scored by a model that never saw it.

We then break MAE / MAPE / median-abs-error / signed BIAS (mean(pred-actual)) down by:
  - postcode AREA (SW/W/NW/...) + prime vs non-prime
  - BEDS (0,1,2,3,4,5+)
  - PRICE BAND (<2k / 2-3k / 3-5k / 5-10k / >10k pcm)
  - PROPERTY TYPE (flat/house/...)
  - sqft present vs SYNTHETICALLY missing (estimateSqft(beds) prior)

Cross-check: the postcode_normalized->fillna('SW3') quirk and its effect on prime.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

import retrain_canonical as rc
import rental_price_models_v20 as v20

DB = 'output/rentals.db'
PRIME_POSTCODES = v20.PRIME_POSTCODES  # ['SW1','SW3','SW7','SW10','W1','W8','W11','NW3','NW8']

# content.js estimateSqft(beds) prior (the size-guess used when sqft is missing)
ESTIMATE_SQFT = {0: 350, 1: 500, 2: 750, 3: 1000, 4: 1300, 5: 1600}
def estimate_sqft(beds):
    b = int(round(beds)) if pd.notna(beds) else 1
    if b >= 5:
        return ESTIMATE_SQFT[5]
    return ESTIMATE_SQFT.get(b, 750)


def build_Xy(df, feat_cols):
    df = df.copy()
    # Zero-fill any declared-but-absent column (parity with canonical_predict.build_features:
    # a small slice may not emit every one-hot dummy, e.g. type_penthouse).
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feat_cols].copy()
    for c in X.columns:
        if X[c].dtype == 'bool':
            X[c] = X[c].astype(int)
        elif X[c].dtype == 'object':
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)
    return X


def oof_predict(X, y):
    """Out-of-fold predictions, same fold scheme as retrain_canonical.cv_metrics."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    yl = np.log1p(y)
    oof = np.zeros(len(X))
    for tr, va in kf.split(X):
        m = v20.build_xgboost()
        m.fit(X.iloc[tr], yl.iloc[tr])
        oof[va] = np.expm1(m.predict(X.iloc[va]))
    return oof


def seg_metrics(g):
    actual = g['actual'].values
    pred = g['pred'].values
    err = pred - actual
    ape = np.abs(err) / actual
    return pd.Series({
        'n': len(g),
        'MAE': np.mean(np.abs(err)),
        'MedAE': np.median(np.abs(err)),
        'MAPE%': np.mean(ape) * 100,
        'MedAPE%': np.median(ape) * 100,
        'BIAS': np.mean(err),                 # +ve = over-predict, -ve = under-predict
        'BIAS%': np.mean(err / actual) * 100, # signed pct
        'med_actual': np.median(actual),
    })


def show(title, df, by):
    print(f"\n{'='*92}\n{title}\n{'='*92}")
    out = df.groupby(by, dropna=False).apply(seg_metrics)
    out = out.sort_values('n', ascending=False)
    with pd.option_context('display.float_format', lambda v: f'{v:,.1f}', 'display.width', 200):
        print(out.to_string())
    return out


def main():
    print("Loading + engineering (identical to retrain_canonical) ...")
    df = rc.load_recency_independent(DB)

    # Keep raw segmenting columns BEFORE engineering mangles them.
    raw = pd.DataFrame({
        'raw_postcode': df['postcode'],
        'raw_postcode_norm': df['postcode_normalized'],
        'raw_postcode_norm_isnull': df['postcode_normalized'].isnull(),
        'raw_property_type': df['property_type_std'].fillna(df['property_type']).fillna('unknown').str.lower(),
        'raw_beds': df['bedrooms'],
        'raw_sqft': df['size_sqft'],
        'raw_source': df['source'],
    }).reset_index(drop=True)

    df = v20.engineer_features_v20(df).reset_index(drop=True)
    feat_cols = v20.get_feature_columns_v20(df)
    X = build_Xy(df, feat_cols)
    y = df['price_pcm'].reset_index(drop=True)

    print(f"\nFeatures: {len(feat_cols)}  Samples: {len(X)}")

    print("\n[OUT-OF-FOLD prediction, 5-fold random_state=42]")
    pred = oof_predict(X, y)

    # Overall sanity (should match retrain_canonical cv ~R2 .79 MAE ~1500)
    err = pred - y.values
    print(f"  OVERALL  n={len(y)}  MAE=£{np.mean(np.abs(err)):,.0f}  "
          f"MedAE=£{np.median(np.abs(err)):,.0f}  "
          f"MAPE={np.mean(np.abs(err)/y.values)*100:.1f}%  "
          f"MedAPE={np.median(np.abs(err)/y.values)*100:.1f}%  "
          f"BIAS=£{np.mean(err):,.0f}")

    # Assemble the scoring frame with raw segmenting cols + engineered area/prime
    S = pd.DataFrame({
        'actual': y.values,
        'pred': pred,
        'beds': raw['raw_beds'].values,
        'sqft': raw['raw_sqft'].values,
        'ptype': raw['raw_property_type'].values,
        'source': raw['raw_source'].values,
        'pc_norm_null': raw['raw_postcode_norm_isnull'].values,
        # engineered:
        'postcode_district': df['postcode_district'].values,
        'postcode_area': df['postcode_area'].values,
        'is_prime_postcode': df['is_prime_postcode'].values,
    })

    # --- Beds band ---
    def beds_band(b):
        b = int(round(b)) if pd.notna(b) else -1
        return '5+' if b >= 5 else str(max(b, 0))
    S['beds_band'] = S['beds'].apply(beds_band)

    # --- Price band ---
    def price_band(p):
        if p < 2000: return '1 <2k'
        if p < 3000: return '2 2-3k'
        if p < 5000: return '3 3-5k'
        if p < 10000: return '4 5-10k'
        return '5 >10k'
    S['price_band'] = S['actual'].apply(price_band)

    # --- Property type bucket ---
    def ptype_bucket(t):
        t = str(t)
        if 'flat' in t or 'apartment' in t: return 'flat'
        if 'house' in t: return 'house'
        if 'maisonette' in t: return 'maisonette'
        if 'studio' in t: return 'studio'
        if 'mews' in t: return 'mews'
        return 'other/unknown'
    S['ptype_bucket'] = S['ptype'].apply(ptype_bucket)

    S['prime'] = np.where(S['is_prime_postcode'] == 1, 'PRIME', 'non-prime')

    # ===== SEGMENT TABLES =====
    show("BY POSTCODE AREA", S, 'postcode_area')
    show("BY PRIME vs NON-PRIME", S, 'prime')
    show("BY PRIME POSTCODE DISTRICT (prime only)", S[S.is_prime_postcode == 1], 'postcode_district')
    show("BY BEDS", S, 'beds_band')
    show("BY PRICE BAND (actual pcm)", S, 'price_band')
    show("BY PROPERTY TYPE", S, 'ptype_bucket')
    show("BY SOURCE", S, 'source')

    # ===== SW3 fillna quirk cross-check =====
    print(f"\n{'='*92}\nSW3 FILLNA QUIRK CROSS-CHECK\n{'='*92}")
    n_null = int(S['pc_norm_null'].sum())
    print(f"postcode_normalized NULL rows (fillna->'SW3'): {n_null} / {len(S)} "
          f"({n_null/len(S)*100:.1f}%)")
    print(f"Rows whose engineered postcode_district == 'SW3': "
          f"{int((S.postcode_district=='SW3').sum())}")
    print("\nError on TRUE-SW3 (had a real SW3 postcode) vs FILLNA-SW3 (postcode_norm was null):")
    sw3 = S[S.postcode_district == 'SW3'].copy()
    sw3['sw3_kind'] = np.where(sw3['pc_norm_null'], 'FILLNA (null->SW3)', 'TRUE SW3')
    show("  SW3 rows split by origin", sw3, 'sw3_kind')

    # ===== Synthetic missing-sqft experiment =====
    # Refit the model NOT changing the artifact -- a fresh CV where each held-out
    # row's size_sqft is REPLACED by estimateSqft(beds) at predict time, mirroring
    # how the extension serves chestertons (no sqft). Train on TRUE sqft (as prod),
    # predict on the GUESSED sqft, measure the induced error vs true price.
    print(f"\n{'='*92}\nSYNTHETIC MISSING-SQFT (estimateSqft(beds) prior) -- induced error\n{'='*92}")
    print("Train on real sqft (as prod); at predict time replace held-out sqft with "
          "estimateSqft(beds), re-engineer, score vs true price.")

    # Build a raw frame we can re-engineer with swapped sqft.
    base = rc.load_recency_independent(DB).reset_index(drop=True)
    y2 = base['price_pcm'].reset_index(drop=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    yl = np.log1p(y2)

    # Train folds on TRUE-sqft engineered features.
    df_true = v20.engineer_features_v20(base.copy()).reset_index(drop=True)
    fc = v20.get_feature_columns_v20(df_true)
    X_true = build_Xy(df_true, fc)

    oof_guess = np.zeros(len(base))
    for tr, va in kf.split(base):
        m = v20.build_xgboost()
        m.fit(X_true.iloc[tr], yl.iloc[tr])
        # Re-engineer the validation rows with GUESSED sqft.
        vb = base.iloc[va].copy()
        vb['size_sqft'] = vb['bedrooms'].apply(estimate_sqft)
        vb_eng = v20.engineer_features_v20(vb)
        Xv = build_Xy(vb_eng, fc).reindex(columns=fc, fill_value=0)
        oof_guess[va] = np.expm1(m.predict(Xv))

    G = pd.DataFrame({
        'actual': y2.values,
        'pred_guess': oof_guess,
        'pred_true': pred if len(pred) == len(y2) else np.nan,
        'beds': base['bedrooms'].values,
        'true_sqft': base['size_sqft'].values,
        'guess_sqft': base['bedrooms'].apply(estimate_sqft).values,
        'postcode_district': df_true['postcode_district'].values,
        'is_prime_postcode': df_true['is_prime_postcode'].values,
    })
    G['prime'] = np.where(G['is_prime_postcode'] == 1, 'PRIME', 'non-prime')
    err_g = G['pred_guess'].values - G['actual'].values
    print(f"  GUESSED-SQFT OVERALL  MAE=£{np.mean(np.abs(err_g)):,.0f}  "
          f"MAPE={np.mean(np.abs(err_g)/G['actual'].values)*100:.1f}%  "
          f"BIAS=£{np.mean(err_g):,.0f}  (true-sqft BIAS was £{np.mean(err):,.0f})")
    print(f"  Mean guess_sqft - true_sqft = {(G['guess_sqft']-G['true_sqft']).mean():,.0f} sqft "
          f"(prime: {(G[G.is_prime_postcode==1]['guess_sqft']-G[G.is_prime_postcode==1]['true_sqft']).mean():,.0f})")

    def gseg(g):
        a = g['actual'].values
        eg = g['pred_guess'].values - a
        return pd.Series({
            'n': len(g),
            'guess_MAPE%': np.mean(np.abs(eg)/a)*100,
            'guess_BIAS': np.mean(eg),
            'guess_BIAS%': np.mean(eg/a)*100,
            'mean_sqft_gap': (g['guess_sqft']-g['true_sqft']).mean(),
        })
    print("\n  By beds (guessed sqft):")
    with pd.option_context('display.float_format', lambda v: f'{v:,.1f}'):
        gb = G.copy()
        gb['beds_band'] = gb['beds'].apply(lambda b: '5+' if (pd.notna(b) and b>=5) else str(int(b)) if pd.notna(b) else 'na')
        print(gb.groupby('beds_band').apply(gseg).to_string())
    print("\n  By prime (guessed sqft):")
    with pd.option_context('display.float_format', lambda v: f'{v:,.1f}'):
        print(G.groupby('prime').apply(gseg).to_string())

    print("\nDONE.")


if __name__ == '__main__':
    main()
