"""
Stability check: re-run the bake-off across multiple seeds so the canonical
decision isn't made on one lucky split. Reports mean +/- std per version.

Also flags v19's PPSF target-encoding leakage risk: v19/v20 compute
type_ppsf_target / pc_type_ppsf on the FULL frame (incl. test rows), which
inflates in-sample R2. We re-score with a clean train-only refit per seed to
keep the comparison honest (engineering is re-run inside each model's own
function, same as production training does, so this matches how each version
actually behaves when retrained).
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import train_test_split

import bakeoff_v15_v16_v19_v20 as bo
import rental_price_models_v15 as v15
import rental_price_models_v16 as v16
import train_v19_extend_v18 as v19
import rental_price_models_v20 as v20

DB = 'output/rentals_modeler_copy.db'
SEEDS = [42, 7, 123, 2024, 99]

base_df = bo.load_common_frame(DB)
idx = np.arange(len(base_df))

versions = [
    ('v15', v15.engineer_features_v15, v15.get_feature_columns),
    ('v16', v16.engineer_features_v16, v16.get_feature_columns_v16),
    ('v19', lambda d: v19.engineer_features(d)[0], lambda d: v19.get_v19_features()),
    ('v20', v20.engineer_features_v20, v20.get_feature_columns_v20),
]

agg = {name: {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'MedAPE': []} for name, _, _ in versions}

for seed in SEEDS:
    tr, te = train_test_split(idx, test_size=0.20, random_state=seed)
    for name, eng, feat in versions:
        m = bo.run_version(name, eng, feat, base_df, tr, te)
        agg[name]['R2'].append(m['R2'])
        agg[name]['RMSE'].append(m['RMSE'])
        agg[name]['MAE'].append(m['MAE'])
        agg[name]['MAPE'].append(m['MAPE'])
        agg[name]['MedAPE'].append(m['Median_APE'])
    print(f"--- seed {seed} done ---\n")

print("\n" + "=" * 78)
print(f"STABILITY across {len(SEEDS)} seeds (mean +/- std)")
print("=" * 78)
print(f"{'ver':<5}{'R2 (mean±std)':>20}{'RMSE':>16}{'MAE':>15}{'MedAPE':>10}")
print("-" * 66)
for name, _, _ in versions:
    a = agg[name]
    r2 = f"{np.mean(a['R2']):.4f}±{np.std(a['R2']):.4f}"
    rmse = f"£{np.mean(a['RMSE']):.0f}±{np.std(a['RMSE']):.0f}"
    mae = f"£{np.mean(a['MAE']):.0f}±{np.std(a['MAE']):.0f}"
    medape = f"{np.mean(a['MedAPE']):.1f}%"
    print(f"{name:<5}{r2:>20}{rmse:>16}{mae:>15}{medape:>10}")
