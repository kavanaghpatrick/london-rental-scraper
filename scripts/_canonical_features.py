"""
_canonical_features.py — resolve the canonical feature-engineering pipeline.

The canonical model version (v15 vs v20 vs …) is the modeler's call, and it is
recorded in ONE place: `retrain_canonical.py` (its `CANON_VERSION` constant and the
`rental_price_models_v<NN>` module it imports). Prediction-side scripts must NOT
hardcode `import rental_price_models_v20` — if the canonical version changes, that
would silently build the wrong feature schema for the served model.

This resolver derives, at runtime, the SAME feature module + functions that
`retrain_canonical.py` used to train the canonical artifacts. So whatever version
the modeler/user blesses, generate_predictions.py and predict_one.py follow it with
no code change.

Returns (engineer_features, get_feature_columns, version_str).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_feature_pipeline():
    """Return (engineer_features_fn, get_feature_columns_fn, version) from the
    canonical trainer's chosen model module — version-agnostic.
    """
    import retrain_canonical as rc

    version = getattr(rc, "CANON_VERSION", None)

    # retrain_canonical imports the chosen model module and aliases it (currently
    # `import rental_price_models_v20 as v20`). Find that module object regardless of
    # the alias by matching a `rental_price_models_*` module bound in its namespace.
    model_mod = None
    for name in dir(rc):
        obj = getattr(rc, name)
        mod_name = getattr(obj, "__name__", "")
        if isinstance(mod_name, str) and mod_name.startswith("rental_price_models_"):
            model_mod = obj
            if version is None:
                version = mod_name.replace("rental_price_models_", "")
            break

    if model_mod is None:
        raise RuntimeError(
            "Could not resolve the canonical feature module from retrain_canonical.py. "
            "Expected an `import rental_price_models_v<NN>` there."
        )

    # Find the engineer_features_* and get_feature_columns_* functions on that module
    # without assuming a specific version suffix.
    engineer = None
    get_cols = None
    for attr in dir(model_mod):
        if attr.startswith("engineer_features"):
            engineer = getattr(model_mod, attr)
        elif attr.startswith("get_feature_columns"):
            get_cols = getattr(model_mod, attr)

    if engineer is None:
        raise RuntimeError(
            f"{model_mod.__name__} has no engineer_features* function — cannot build features."
        )

    return engineer, get_cols, version


def resolve_model_module():
    """Return the canonical model module object (e.g. rental_price_models_v20) that
    retrain_canonical.py used — for callers that need export_to_chrome etc."""
    import retrain_canonical as rc

    for name in dir(rc):
        obj = getattr(rc, name)
        mod_name = getattr(obj, "__name__", "")
        if isinstance(mod_name, str) and mod_name.startswith("rental_price_models_"):
            return obj
    raise RuntimeError("Could not resolve the canonical model module from retrain_canonical.py.")

