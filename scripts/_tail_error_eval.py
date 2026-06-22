"""
_tail_error_eval.py — Tail-accuracy harness for the canonical v20 rental model.

PURPOSE
-------
The v20 model UNDER-prices the expensive end (>£10k) and OVER-prices the cheap end
(<£2k). This harness quantifies that signed bias by PRICE BAND and by BEDS using
OUT-OF-FOLD predictions, so the "tail accuracy" fixes can be measured BEFORE vs AFTER
on an identical, leakage-free protocol.

PROTOCOL (mirrors retrain_canonical.cv_metrics EXACTLY so the numbers are the real
training-path numbers, not a different CV):
    - Load via retrain_canonical.load_recency_independent(<db>)  (no is_active filter)
    - Engineer via v20.engineer_features_v20 + v20.get_feature_columns_v20
    - X-prep identical to retrain_canonical.main(): bool->int, object->to_numeric,
      fillna(0)
    - KFold(n_splits=5, shuffle=True, random_state=42)
    - target = log1p(price_pcm); model = v20.build_xgboost(); preds = expm1(pred)
    - OUT-OF-FOLD: every row is predicted by a model that did NOT see it in training.

LEAKAGE NOTE
------------
engineer_features_v20 computes postcode_freq / postcode_area_freq as
(count in THIS frame / len(frame)). Here we engineer the FULL frame once (exactly as
the trainer does in retrain_canonical.main, which also engineers the full df before the
KFold split). This matches the production training path bit-for-bit, so the OOF numbers
this harness reports ARE the model's honest CV behaviour. We do NOT inject single-row
inference maps (that is a serving-time concern, out of scope for this measurement).

Run (re-runnable for BEFORE and AFTER passes):
    python3 scripts/_tail_error_eval.py
    python3 scripts/_tail_error_eval.py --db output/rentals_modeler_copy.db
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Run from the repo root regardless of cwd so the v20/retrain imports resolve.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import retrain_canonical as rc          # noqa: E402
import rental_price_models_v20 as v20   # noqa: E402

# Price bands (left-closed, right-open) per the task spec.
PRICE_BANDS = [
    (0, 2000, '<2k'),
    (2000, 5000, '2-5k'),
    (5000, 10000, '5-10k'),
    (10000, 20000, '10-20k'),
    (20000, float('inf'), '>20k'),
]


def compute_oof(db_path):
    """Return (df, y, oof_preds) where oof_preds[i] is the OOF prediction for row i.

    Mirrors retrain_canonical.cv_metrics + retrain_canonical.main X-prep exactly.
    """
    df = rc.load_recency_independent(db_path)
    df = v20.engineer_features_v20(df)
    feat_cols = v20.get_feature_columns_v20(df)

    X = df[feat_cols].copy()
    for c in X.columns:
        if X[c].dtype == 'bool':
            X[c] = X[c].astype(int)
        elif X[c].dtype == 'object':
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)
    y = df['price_pcm'].reset_index(drop=True)

    yl = np.log1p(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(X), np.nan, dtype=float)
    for tr, va in kf.split(X):
        m = v20.build_xgboost(feat_cols)
        m.fit(X.iloc[tr], yl.iloc[tr])
        oof[va] = np.expm1(m.predict(X.iloc[va]))

    assert not np.isnan(oof).any(), "Some rows never received an OOF prediction"
    return df.reset_index(drop=True), y.values, oof, len(feat_cols)


def _row_stats(actual, preds):
    """Signed bias (mean pred-actual, abs + %), MAPE, count for an array slice."""
    n = len(actual)
    if n == 0:
        return None
    signed = preds - actual
    signed_bias = float(np.mean(signed))
    signed_bias_pct = float(np.mean(signed / actual) * 100)   # mean of per-row signed %
    mape = float(np.mean(np.abs(signed / actual)) * 100)
    return {'n': n, 'bias_gbp': signed_bias, 'bias_pct': signed_bias_pct, 'mape': mape}


def band_table(actual, preds):
    rows = []
    for low, high, label in PRICE_BANDS:
        mask = (actual >= low) & (actual < high)
        st = _row_stats(actual[mask], preds[mask])
        if st is None:
            rows.append((label, 0, None, None, None))
        else:
            rows.append((label, st['n'], st['bias_pct'], st['bias_gbp'], st['mape']))
    return rows


def beds_table(df, actual, preds):
    rows = []
    beds = df['bedrooms'].values
    # Group beds: 0,1,2,3,4,5+
    groups = [(0, '0'), (1, '1'), (2, '2'), (3, '3'), (4, '4')]
    for b, label in groups:
        mask = beds == b
        st = _row_stats(actual[mask], preds[mask])
        if st is None:
            rows.append((label, 0, None, None, None))
        else:
            rows.append((label, st['n'], st['bias_pct'], st['bias_gbp'], st['mape']))
    mask = beds >= 5
    st = _row_stats(actual[mask], preds[mask])
    if st is None:
        rows.append(('5+', 0, None, None, None))
    else:
        rows.append(('5+', st['n'], st['bias_pct'], st['bias_gbp'], st['mape']))
    return rows


def _fmt(rows, key_label):
    lines = []
    header = f"  {key_label:<8} {'n':>6} {'signed bias %':>14} {'signed bias £':>16} {'MAPE %':>9}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for label, n, bias_pct, bias_gbp, mape in rows:
        if n == 0:
            lines.append(f"  {label:<8} {n:>6} {'--':>14} {'--':>16} {'--':>9}")
        else:
            lines.append(
                f"  {label:<8} {n:>6} {bias_pct:>+13.1f}% £{bias_gbp:>+13,.0f}  {mape:>8.1f}%"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='output/rentals_modeler_copy.db')
    args = ap.parse_args()

    db = args.db
    if not Path(db).is_absolute():
        db = str(_REPO / db)

    print("=" * 74)
    print("TAIL-ERROR HARNESS (OOF, KFold5 shuffle rs=42, log1p target) — v20 canonical")
    print(f"DB: {db}")
    print("=" * 74)

    df, actual, preds, n_feats = compute_oof(db)
    actual = np.asarray(actual, dtype=float)
    preds = np.asarray(preds, dtype=float)

    # ── Overall ──────────────────────────────────────────────────────────────
    r2 = r2_score(actual, preds)
    mae = mean_absolute_error(actual, preds)
    ape = np.abs((actual - preds) / actual)
    mape = float(np.mean(ape) * 100)
    medape = float(np.median(ape) * 100)
    overall = (f"n={len(actual)}  features={n_feats}  R2={r2:.4f}  MAE=£{mae:,.0f}  "
               f"MAPE={mape:.1f}%  MedAPE={medape:.1f}%")

    bt = band_table(actual, preds)
    bdt = beds_table(df, actual, preds)

    print("\n[OVERALL]")
    print("  " + overall)

    print("\n[SIGNED BIAS + MAPE by PRICE BAND]  (signed = pred - actual; +%=over-price, -%=under-price)")
    print(_fmt(bt, 'band'))

    print("\n[SIGNED BIAS + MAPE by BEDS]")
    print(_fmt(bdt, 'beds'))

    # Machine-greppable single-line markers for the AFTER comparison.
    print("\n[MACHINE]")
    print(f"OVERALL\t{overall}")
    for label, n, bias_pct, bias_gbp, mape_b in bt:
        if n:
            print(f"BAND\t{label}\tn={n}\tbias_pct={bias_pct:+.1f}\tbias_gbp={bias_gbp:+.0f}\tmape={mape_b:.1f}")
    for label, n, bias_pct, bias_gbp, mape_b in bdt:
        if n:
            print(f"BEDS\t{label}\tn={n}\tbias_pct={bias_pct:+.1f}\tbias_gbp={bias_gbp:+.0f}\tmape={mape_b:.1f}")

    return overall, bt, bdt


if __name__ == '__main__':
    main()
