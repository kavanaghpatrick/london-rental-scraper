# Model evaluation: where is the v20 rent model strong / where does it fail?

**Trigger:** extension showed an SW10 (Chelsea) 2bed/2bath at asking £4,500 as MODEL ESTIMATE £2,373 / "+89.6% OVERPRICED" — clearly the MODEL under-predicting a prime 2bed, not the listing being overpriced. Goal: systematically map strong vs weak, anchored on this case.

**Model:** v20 canonical (output/rental_model_canonical.pkl, 135 features). R²≈0.792, MAE≈£1,507 (full-set retrain). sqft is a top feature. Predict via `canonical_predict.predict_one(**fields)`.

## Lead's initial findings (start here — these are the threads)
The £2,373 has THREE contributing causes, found by reproducing locally:

1. **The model is FINE at realistic sizes.** SW10 2bed/2bath:
   950sqft→£4,175, 1050sqft→£4,641 — right next to the £4,500 asking. So the model is NOT fundamentally broken on prime Chelsea. The problem is the INPUTS + serving, not (mostly) the core model on this case.

2. **Missing sqft → size GUESSED too small.** chestertons exposes no sqft, so content.js `estimateSqft(beds)` (content.js:1428) maps 2 beds → **750 sqft** via `{0:350,1:500,2:750,3:1000,4:1300,5:1600}`. A real Chelsea 2bed is ~900-1050 sqft. 750 is low → drags the estimate down. **Evaluate: is estimateSqft(beds) systematically too small (esp. prime), and how much does it bias predictions when sqft is missing?**

3. **A SERVING DIVERGENCE (likely the biggest finding).** Even at 750 sqft, canonical Python predicts **£3,437** — but the extension SHOWED **£2,373**, a ~£1,000 (31%) gap. The extension serves via /api/predict + the JS predictor (xgboost.js), whose JS↔Python parity is only VERIFIED on 9 golden samples. On real/arbitrary inputs the JS predictor may diverge materially from the canonical Python. **Evaluate: reproduce the served value vs canonical Python on this exact input + a broad sample; quantify the JS↔Python parity gap on REAL inputs (not just the 9 golden). If it's large, the user is seeing a predictor that diverges from the "true" model.**

4. **MODEL-QUALITY RED FLAG — non-monotonic in size.** SW10 2bed/2bath: 450sqft→£4,470 but 650sqft→£3,375 and 750→£3,437. A SMALLER flat predicting HIGHER than a larger one is wrong — likely the is_tiny flag / ppsf or size-per-bed interactions misbehaving at small sizes. **Evaluate: is the model non-monotonic in size across segments? Where does it break? Is the tiny-property / ppsf handling sane?**

## Evaluation to run (each owner takes a thread)
1. **Serving parity on real inputs:** dissect £2,373(served) vs £3,437(Python). Reproduce the JS predictor output (node) on this exact input; compare to canonical Python; find the diverging features; then sample N real listings and quantify the JS↔Python £-gap distribution (the 9-sample golden parity is NOT representative). This is what the USER actually sees.
2. **Segment error map (strong vs weak):** compute MAE/MAPE/bias by postcode area, beds, price band, property type, and sqft-present-vs-estimated — on a held-out or representative slice of output/rentals.db (use the same load/clean/feature pipeline as retrain). Identify where the model is trustworthy vs not. Watch the [[postcode-normalized-fillna-quirk]] (SW3 over-rep) + prime-central behavior.
3. **Missing-sqft / estimateSqft behavior:** how often is sqft missing per source (esp. chestertons)? How biased is estimateSqft(beds) vs real sqft (compute real avg sqft per beds per area from the DB)? Quantify the prediction bias introduced by the size-guess. Recommend a better size prior (area×beds median?).
4. **Feature-behavior / non-monotonicity:** confirm + characterize the non-monotonic size response; check other key features (beds, baths, distance, prestige) for sane monotonic behavior; flag any feature interactions that misfire (tiny/huge flags, ppsf).

## Output
A "where the model is strong / where it fails" report: the segment error map, the serving-parity gap, the size-estimation bias, the non-monotonicity, + prioritized recommendations (serving-parity fix, estimateSqft improvement, retrain/feature priorities, or guardrails like suppressing/flagging low-confidence estimates when sqft is missing). This is ANALYSIS — propose fixes, don't necessarily implement in this pass.

## Constraints
- Use canonical_predict + output/rental_model_canonical.pkl + output/rentals.db. Don't retrain/modify the model in the eval pass.
- The JS predictor lives in chrome-extension/xgboost.js (+ node _fixture_diff.mjs harness to run it). Don't edit it during eval.
