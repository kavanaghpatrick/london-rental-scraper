"""for_sale.sale_predict — the lazy-load PREDICT convenience for the for-sale model (Inc3 §3).

This is the Python serving ergonomics layer over the baseline sale model: a single
zero-DB, network-free entrypoint, ``predict_one_default``, that lazy-loads the committed
``output/sale_model.pkl`` (cached), wires INFERENCE MODE (the baked district / postcode-area
frequency maps from ``output/sale_model_inference.json`` — the BLOCKER-1 single-row-degeneracy
fix), and returns a RICHER dict than a bare float so Inc4's serving route has the UX signals
it needs (low_confidence / estimated_size / district).

INC4 CONTRACT (documented here so Inc4's request layer knows the input union)
-----------------------------------------------------------------------------
The fields ``build_features`` reads = {postcode, bedrooms, bathrooms, size_sqft,
property_type, address, is_new_build, latitude, longitude, price_qualifier}. Any missing
column is zero-aligned by ``reindex(fill_value=0.0)``; every Inc3 feature degrades to a
neutral constant when its source field is absent. The bare-float ``predict_one`` in
sale_price_model is what Inc4's JS will mirror; ``predict_one_default`` is the Python-side
convenience (Inc4 wires the actual route, not us).

ISOLATION CONTRACT (enforced by tests/test_for_sale_inc3_model.py guards)
-------------------------------------------------------------------------
This module imports ONLY for_sale.sale_price_model (+ for_sale.sale_retrain for the
artifact paths, when available) + stdlib / numpy. It NEVER imports the rental chain
(rental_price_models_v20 / canonical_predict / retrain_canonical / the rental sidecar
generators) — the inference-mode plumbing and the n_features_in_ drift guard are
RE-IMPLEMENTED BY VALUE, mirroring canonical_predict.load_canonical(:189) shape only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Sibling for-sale modules ONLY — never the rental chain. sale_price_model carries the
# feature builder, the artifact loader, the district parser, the sale-magnitude bounds and
# the default pickle paths; all re-implemented tenure-agnostic, never imported from rental.
from for_sale import sale_price_model
from for_sale.sale_price_model import (
    DEFAULT_MODEL_PATH,
    DEFAULT_FEATURES_PATH,
    MAX_SALE_PRICE,
    MIN_SALE_PRICE,
    OUTPUT_DIR,
    build_features,
    load_model,
    postcode_to_district,
)

# The baked-frequency sidecar path. Prefer the canonical constant exported by the training
# harness (for_sale.sale_retrain) so predict + retrain agree byte-for-byte on the location;
# fall back to the derived default if the harness module is not importable yet (e.g. this
# module is exercised in isolation by the inference-mode build_features test, which never
# touches sale_retrain). The fallback is the SAME path the harness writes (output/), so the
# two can never silently diverge.
try:  # pragma: no cover - import-availability branch
    from for_sale.sale_retrain import INFERENCE_PATH as _DEFAULT_INFERENCE_PATH
except Exception:  # pragma: no cover - harness not present in this import context
    _DEFAULT_INFERENCE_PATH = OUTPUT_DIR / "sale_model_inference.json"

# Public alias matching the spec §3.2 signature default name.
INFERENCE_PATH = _DEFAULT_INFERENCE_PATH

# ── Module-level lazy-load cache, keyed by (model_path, features_path) ────────────────────
# Mirrors canonical_predict's single-load discipline: the (possibly large) pickle pair is
# loaded once per distinct path pair and reused. The drift guard (n_features_in_ vs
# len(feature_cols)) runs ON LOAD, so a freshly-loaded — or cache-cleared — pair is always
# re-validated. Tests clear this between cases (predict_mod._CACHE.clear()).
_CACHE: dict[tuple[str, str], tuple[object, list[str]]] = {}


def _load_cached(model_path: Path | str, features_path: Path | str):
    """Lazy-load (model, feature_cols) for a path pair, cached, with the drift guard.

    On (cache-missing) load it ASSERTS model.n_features_in_ == len(feature_cols) and raises
    ValueError on a mismatch — mirroring canonical_predict.load_canonical(:189) so a drifted
    features file fails LOUDLY instead of silently mis-predicting against a wrong-width matrix.
    """
    key = (str(model_path), str(features_path))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    model, feature_cols = load_model(model_path, features_path)
    feature_cols = list(feature_cols)

    # Drift guard (mirror canonical_predict.load_canonical :189): the pickled model's input
    # width MUST equal the feature-column list length, else the reindex/predict would line up
    # the wrong columns. n_features_in_ is set by sklearn/xgboost at fit time.
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(feature_cols):
        raise ValueError(
            "sale model feature drift: model.n_features_in_="
            f"{int(n_in)} != len(feature_cols)={len(feature_cols)} "
            f"(model={model_path}, features={features_path})"
        )

    _CACHE[key] = (model, feature_cols)
    return model, feature_cols


def _load_inference_maps(inference_path: Path | str):
    """Load the baked freq sidecar if present → (freq_map, freq_default, area_map, area_default).

    Returns (None, None, None, None) when the sidecar is absent so callers fall back to the
    per-frame frequency path (still finite; used by tests that don't write the sidecar). The
    sidecar's top-level *_default is a NUMERIC value == min(map) (the _assert_default_shape
    invariant, mirrored by value), so it is passed straight through as freq_default.
    """
    try:
        path = Path(inference_path)
        if not path.exists():
            return None, None, None, None
        stats = json.loads(path.read_text())
    except Exception:
        return None, None, None, None

    freq_map = stats.get("district_freq")
    freq_default = stats.get("district_freq_default")
    area_freq_map = stats.get("postcode_area_freq")
    area_freq_default = stats.get("postcode_area_freq_default")
    return freq_map, freq_default, area_freq_map, area_freq_default


def predict_one_default(
    *,
    postcode: str | None = None,
    bedrooms: int = 1,
    bathrooms: int = 1,
    size_sqft: float = 0.0,
    property_type: str = "flat",
    address: str = "",
    is_new_build: int = 0,
    latitude=None,
    longitude=None,
    price_qualifier: str = "",
    model_path: Path | str = DEFAULT_MODEL_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
    inference_path: Path | str = INFERENCE_PATH,
) -> dict:
    """Predict a fair SALE value (£) for a single property from the committed sale artifact.

    Lazy-loads (model, feature_cols) via the cached loader (with the n_features_in_ drift
    guard), builds the feature row in INFERENCE MODE when the baked sidecar is present (so
    district_freq / postcode_area_freq use the TRAINING distribution, not the degenerate
    single-row per-frame 1.0 — the BLOCKER-1 fix), reindexes to the trained column order,
    and inverts the log1p target with np.expm1.

    Returns a dict (Inc4 serving ergonomics):
        {"predicted_price": float,   # finite £, clamped into [MIN_SALE_PRICE, MAX_SALE_PRICE]
         "low_confidence": bool,     # True if size missing/estimated OR district == "UNKNOWN"
         "district": str,            # outward-code district ('UNKNOWN' when postcode absent)
         "estimated_size": bool}     # True when size_sqft was missing / non-positive

    Tolerates postcode=None → district 'UNKNOWN' (its own neutral low-freq bucket) → a finite
    price with low_confidence=True; never crashes on a sparse / coordless row.
    """
    model, feature_cols = _load_cached(model_path, features_path)

    # District is derived for the returned dict + the low-confidence signal. The builder
    # parses it again internally; this is the public-facing copy (mirrors the loader path).
    district = postcode_to_district(postcode)

    # A missing / non-positive size is the dominant UX low-confidence trigger: the model then
    # imputes a neutral median size (sale_price_model.build_features), so the prediction is an
    # estimate, not a measured-size read.
    try:
        size_val = float(size_sqft)
    except (TypeError, ValueError):
        size_val = 0.0
    estimated_size = not (size_val > 0)

    # Assemble the single inference row. Only positive sizes are passed through; a 0/missing
    # size is left out so build_features applies its own neutral median fallback (rather than
    # treating 0 sqft as a real measurement).
    row: dict = {
        "postcode": postcode,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_type": property_type,
        "address": address,
        "is_new_build": is_new_build,
        "price_qualifier": price_qualifier,
    }
    if not estimated_size:
        row["size_sqft"] = size_val
    if latitude is not None:
        row["latitude"] = latitude
    if longitude is not None:
        row["longitude"] = longitude

    # INFERENCE MODE wiring: use the baked training-distribution freq maps when the sidecar
    # exists (kills the single-row district_freq==1.0 degeneracy); otherwise fall back to the
    # per-frame path (still finite — used only by tests that don't write the sidecar).
    freq_map, freq_default, area_freq_map, area_freq_default = _load_inference_maps(inference_path)
    if freq_map is not None:
        X, built_cols = build_features(
            [row],
            inference=True,
            freq_map=freq_map,
            freq_default=freq_default,
            area_freq_map=area_freq_map,
            area_freq_default=area_freq_default,
        )
    else:
        X, built_cols = build_features([row])

    # Align to the trained feature order; zero-fill any column the trainer had that this
    # sparse row lacks (the rental serving pattern, re-implemented by value).
    X = X.reindex(columns=feature_cols, fill_value=0.0).astype(float)
    pred_log = model.predict(X)
    price = float(np.expm1(pred_log)[0])

    # Clamp into the sale-magnitude sanity band (defends Inc4's UI against a degenerate row
    # producing an out-of-range £; mirrors the bounds the data-quality filter enforces).
    if not np.isfinite(price):
        price = float(MIN_SALE_PRICE)
    price = float(min(max(price, MIN_SALE_PRICE), MAX_SALE_PRICE))

    low_confidence = bool(estimated_size or district == "UNKNOWN")

    return {
        "predicted_price": price,
        "low_confidence": low_confidence,
        "district": district,
        "estimated_size": bool(estimated_size),
    }
