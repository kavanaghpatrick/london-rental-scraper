"""for_sale.sale_price_model — the BASELINE SALE-PRICE model for the for-sale vertical.

It predicts a property's FAIR SALE VALUE from comparable SCRAPED FOR-SALE ASKING
listings, exactly as rental_price_models_v20 predicts fair rent from comparable
rentals — but on the SALE magnitude (£100k–£50M lump sums) and as a SEPARATE module,
SEPARATE artifact (output/sale_model.pkl), and SEPARATE predict path.

ISOLATION CONTRACT (enforced by guard tests in tests/test_for_sale_sale_model.py)
---------------------------------------------------------------------------------
  * NEVER imports rental_price_models_v20 / canonical_predict. The tenure-agnostic
    feature PATTERNS (size / log_sqft / size_per_bed, postcode→district boundary-
    anchored regex, district & prestige-street encodings, beds / baths / bath_ratio,
    property-type one-hots) are RE-IMPLEMENTED here so a rental retrain can never
    perturb the sale model and vice-versa. (It may reuse a shared SCRAPING primitive,
    but not the rental MODEL chain.)
  * The default artifact path is output/sale_model.pkl, never the rental pkl.

TARGET = log1p(asking_price). XGBoost regressor (mirrors the rental model family),
trained on a list of for-sale row dicts (from the committed CI sample today; from
output/sales.db once Inc2 wires the spider sale-mode). Sale-magnitude / sale-ppsf
bounds reject rental-scale values.

LEAK-SAFETY: every feature is a function of the PROPERTY only (size / location / type /
beds / baths). There is NO target-derived branch (no ppsf-threshold feature like the
rental is_social_housing leak that was removed). asking_price is the label, never a
feature.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Pure tenure-agnostic primitives (extended type classifiers + distance helpers + the
# frozen neutral distance constant). A sibling for_sale module — NOT the rental chain.
from for_sale import sale_features
from for_sale.sale_features import (
    DEFAULT_CENTER_DISTANCE_KM,
    SALE_CITY_CENTER,
    center_distance_features,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "sale_model.pkl"
DEFAULT_FEATURES_PATH = OUTPUT_DIR / "sale_model_features.pkl"

# ── SALE-magnitude sanity bounds (London prime/greater-London asking prices) ──────
# A sale price is a £100k–£50M+ lump sum — a different magnitude from a rental
# price_pcm — so a rental-scale value (£3,500/mo) is rejected as not-a-sale-price.
MIN_SALE_PRICE = 50_000
MAX_SALE_PRICE = 250_000_000

# Sale price-per-sqft bounds: prime London sale ppsf ~ £500–£3,000/sqft (vs rent
# ~£40–£100/sqft/YEAR). Used to drop absurd training rows; NOT a model feature
# (ppsf embeds the target -> would leak), only a data-quality filter.
SALE_PPSF_MIN = 400
SALE_PPSF_MAX = 5_000

# ── Prime districts (mirror the rental PRIME_POSTCODES set; re-declared, not imported)
PRIME_POSTCODES = ["SW1", "SW3", "SW7", "SW10", "W1", "W8", "W11", "NW3", "NW8"]

# Prestige streets with a tier (mirror the rental prestige semantics; re-declared).
# Tier is a small ordinal premium signal — higher = more prestigious. Re-implemented
# (not imported) to keep the sale vertical decoupled from the rental constants.
PRESTIGE_STREET_TIER = {
    "eaton square": 4, "belgrave square": 4, "the boltons": 4, "wilton crescent": 4,
    "cadogan square": 3, "lowndes square": 3, "cheyne walk": 3, "hans place": 3,
    "sloane street": 3, "chester square": 3, "grosvenor square": 4,
    "onslow square": 2, "lennox gardens": 2, "tregunter road": 2, "pont street": 2,
    "thurloe square": 2, "egerton gardens": 2, "draycott place": 2, "carlyle square": 2,
    "holland park": 1, "elm park gardens": 1, "phillimore gardens": 1,
}

# Boundary-anchored outward-code regex — MIRRORS rental FIX-1: the lookahead terminates
# the district at a space / the inward-code digit / end-of-string so 'SW72ED' -> 'SW7'
# (NOT the greedy 'SW72E') while 'SW1X' / 'W1K' / 'EC1A' are preserved.
_PC_RE = re.compile(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)(?=\s|\d|$)")


def postcode_to_district(postcode: str | None) -> str:
    """Outward-code district from a postcode (boundary-anchored). 'UNKNOWN' when absent
    or unparseable — its own neutral low-frequency bucket, never defaulted to a prime
    district (mirrors the rental FIX-1 'UNKNOWN' sentinel decision)."""
    if not postcode:
        return "UNKNOWN"
    m = _PC_RE.match(str(postcode).strip().upper())
    return m.group(1) if m else "UNKNOWN"


# Alpha-area regex (the leading letters of the outward code, e.g. 'SW' from 'SW3 4TX',
# 'EC' from 'EC1A'). A coarser geographic bucket than the full district — a leak-free
# property-attribute distribution, mirroring the rental postcode-area frequency.
_AREA_RE = re.compile(r"^([A-Z]{1,2})")


def postcode_to_area(postcode: str | None) -> str:
    """Alpha-area of a postcode ('SW3 4TX' -> 'SW', 'W1K' -> 'W'). 'UNKNOWN' when absent or
    unparseable — its own neutral low-frequency bucket (mirrors the district sentinel)."""
    if not postcode:
        return "UNKNOWN"
    m = _AREA_RE.match(str(postcode).strip().upper())
    return m.group(1) if m else "UNKNOWN"


def _prestige_tier(address: str | None) -> int:
    if not address:
        return 0
    al = address.lower()
    best = 0
    for street, tier in PRESTIGE_STREET_TIER.items():
        if street in al and tier > best:
            best = tier
    return best


def is_plausible_sale_price(price: Any) -> bool:
    """True iff `price` is a plausible SALE asking price (not a rent, not zero/None)."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return False
    return MIN_SALE_PRICE <= p <= MAX_SALE_PRICE


def is_plausible_sale_ppsf(ppsf: Any) -> bool:
    try:
        v = float(ppsf)
    except (TypeError, ValueError):
        return False
    return SALE_PPSF_MIN <= v <= SALE_PPSF_MAX


# Property-type one-hot vocabulary (lowercased, mirrors the rental house/flat split).
_HOUSE_TYPES = ("house", "town house", "townhouse", "terraced", "detached", "semi")
_FLAT_TYPES = ("flat", "apartment", "maisonette", "penthouse", "studio")

# The canonical feature column order. Object/text columns (address, postcode, raw type)
# are NOT here — only model-ready numerics.
FEATURE_COLUMNS: tuple[str, ...] = (
    # ── The 18 BASELINE columns (Inc1). Order is FROZEN: it must remain an exact-order
    #    prefix of the expanded Inc3 list so the existing round-trip artifact stays valid.
    "bedrooms",
    "bathrooms",
    "size_sqft",
    "log_sqft",
    "sqrt_sqft",
    "size_per_bed",
    "beds_squared",
    "size_squared",
    "bath_ratio",
    "excess_bathrooms",
    "bed_bath_interaction",
    "is_prime_postcode",
    "prestige_tier",
    "district_freq",
    "is_house",
    "is_flat",
    "is_new_build",
    "size_prime_interaction",
    # ── Inc3 APPENDS the tenure-agnostic families AFTER the baseline 18 (stable order).
    #    Every one degrades to a neutral constant when its source field is absent (1.3).
    # Size family (mirror v20 :705-716).
    "is_tiny",
    "is_huge",
    # Bathroom family (mirror v20 :719-723).
    "has_ensuite_each",
    "high_bathroom_count",
    # Location / postcode-area frequency (mirror v20 :785-818).
    "postcode_area_freq",
    # Property-type expansion (mirror v20 :822-846).
    "is_penthouse",
    "is_maisonette",
    "is_terraced",
    "is_studio",
    # Coordinates / distance family (mirror v20 :940-976) — the headline Inc3 add.
    "center_distance_km",
    "log_center_distance",
    "center_distance_inv",
    # Interactions (mirror v20 :980-1017, magnitude-free shapes only).
    "size_x_central",
    "house_size_interaction",
    "prestige_tier_x_size",
    # Sale-only qualifier encoding (FEATURE-SAFE — never reads the numeric label).
    "price_qualifier_poa",
)


def _classify_type(property_type: str | None) -> tuple[int, int]:
    pt = (property_type or "").lower()
    is_house = int(any(t in pt for t in _HOUSE_TYPES))
    is_flat = int(any(t in pt for t in _FLAT_TYPES))
    if not is_house and not is_flat:
        is_flat = 1  # default to flat (mirrors rental property_type_std fillna('flat'))
    return is_house, is_flat


def build_features(
    rows: list[dict],
    *,
    inference: bool = False,
    freq_map: dict[str, float] | None = None,
    freq_default: float | None = None,
    area_freq_map: dict[str, float] | None = None,
    area_freq_default: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the tenure-agnostic feature matrix from for-sale row dicts.

    Mirrors the rental v20 feature-engineering PATTERNS (re-implemented, not imported)
    and is LEAK-SAFE: asking_price / ppsf / price_pcm are NEVER read here, and no feature
    is derived from the target. Returns (DataFrame, feature_cols).

    FREQUENCY ENCODING — the single inference seam (spec section 1.1):
      * inference=False (DEFAULT, training path): district_freq / postcode_area_freq are
        computed PER-FRAME from the rows passed in (the property-attribute distribution of
        this batch). UNCHANGED behavior for every existing caller — byte-identical.
      * inference=True: the frequencies are looked up from the BAKED `freq_map` /
        `area_freq_map` (with `freq_default` / `area_freq_default` for unseen keys). This
        kills the single-row degeneracy where a one-row frame yields district_freq == 1.0
        (Finding 5 BLOCKER 1). predict_one_default passes the baked maps from the artifact
        sidecar.

    Every Inc3 column degrades to a NEUTRAL constant when its source field is absent (the
    committed 9-field fixture has no latitude/longitude/price_qualifier/is_new_build).
    """
    df = pd.DataFrame(rows).copy()
    n = len(df)

    # Core numerics with safe fills (mirror rental fillna conventions).
    df["bedrooms"] = pd.to_numeric(df.get("bedrooms"), errors="coerce").fillna(1).astype(float)
    df["bathrooms"] = pd.to_numeric(df.get("bathrooms"), errors="coerce").fillna(1).astype(float)
    df["size_sqft"] = pd.to_numeric(df.get("size_sqft"), errors="coerce")
    # Size missing → neutral median (kept out-of-distribution-safe like the rental side).
    if df["size_sqft"].notna().any():
        df["size_sqft"] = df["size_sqft"].fillna(df["size_sqft"].median())
    else:
        df["size_sqft"] = df["size_sqft"].fillna(700.0)
    df["size_sqft"] = df["size_sqft"].astype(float)

    beds_adj = df["bedrooms"].replace(0, 0.5)

    # Size family (mirror rental log_sqft / sqrt / size_per_bed / size_squared).
    df["log_sqft"] = np.log1p(df["size_sqft"])
    df["sqrt_sqft"] = np.sqrt(df["size_sqft"])
    df["size_per_bed"] = df["size_sqft"] / beds_adj
    df["beds_squared"] = df["bedrooms"] ** 2
    df["size_squared"] = df["size_sqft"] ** 2 / 100_000
    # Inc3 size flags (mirror v20 :705-716): tiny / huge thresholds.
    df["is_tiny"] = (df["size_sqft"] < 400).astype(int)
    df["is_huge"] = (df["size_sqft"] >= 3000).astype(int)

    # Bathroom family (mirror rental bath_ratio / excess / interaction).
    df["bath_ratio"] = df["bathrooms"] / beds_adj
    df["excess_bathrooms"] = (df["bathrooms"] - df["bedrooms"]).clip(lower=0)
    df["bed_bath_interaction"] = df["bedrooms"] * df["bathrooms"]
    # Inc3 bathroom flags (mirror v20 :719-723).
    df["has_ensuite_each"] = (df["bath_ratio"] >= 1).astype(int)
    df["high_bathroom_count"] = (df["bathrooms"] >= 4).astype(int)

    # Location family (mirror rental district / is_prime_postcode / prestige).
    df["postcode_district"] = df.get("postcode").apply(postcode_to_district) \
        if "postcode" in df.columns else "UNKNOWN"
    df["postcode_area"] = df.get("postcode").apply(postcode_to_area) \
        if "postcode" in df.columns else "UNKNOWN"
    df["is_prime_postcode"] = df["postcode_district"].apply(
        lambda x: int(any(str(x).startswith(p) for p in PRIME_POSTCODES))
    )
    df["prestige_tier"] = df.get("address", pd.Series([""] * len(df))).apply(_prestige_tier)

    # District + postcode-area frequency encodings (property-attribute distributions, NOT
    # the target — mirror rental postcode_freq / area_freq; leak-free). PER-FRAME at train,
    # BAKED-MAP at inference (the BLOCKER-1 single-row degeneracy fix, Finding 5).
    if inference and freq_map is not None:
        default_d = float(freq_default) if freq_default is not None else 0.0
        df["district_freq"] = df["postcode_district"].map(
            lambda d: float(freq_map.get(d, default_d))
        )
    else:
        freq = df["postcode_district"].value_counts(normalize=True)
        df["district_freq"] = df["postcode_district"].map(freq).fillna(0.0)

    if inference and area_freq_map is not None:
        default_a = float(area_freq_default) if area_freq_default is not None else 0.0
        df["postcode_area_freq"] = df["postcode_area"].map(
            lambda a: float(area_freq_map.get(a, default_a))
        )
    else:
        afreq = df["postcode_area"].value_counts(normalize=True)
        df["postcode_area_freq"] = df["postcode_area"].map(afreq).fillna(0.0)

    # Property-type one-hots (baseline house/flat) + interaction.
    types = df.get("property_type", pd.Series([""] * len(df))).apply(_classify_type)
    df["is_house"] = [t[0] for t in types]
    df["is_flat"] = [t[1] for t in types]
    # Inc3 extended property-type one-hots (mirror v20 :822-846, re-implemented in
    # for_sale.sale_features). A plain Flat / Apartment sets NONE of the four.
    ext = df.get("property_type", pd.Series([""] * len(df))).apply(
        sale_features._classify_type_extended
    )
    df["is_penthouse"] = [t[0] for t in ext]
    df["is_maisonette"] = [t[1] for t in ext]
    df["is_terraced"] = [t[2] for t in ext]
    df["is_studio"] = [t[3] for t in ext]

    df["is_new_build"] = pd.to_numeric(df.get("is_new_build"), errors="coerce").fillna(0).astype(int) \
        if "is_new_build" in df.columns else 0

    # Coordinates / distance family (mirror v20 :940-976) — the headline Inc3 add. Read
    # latitude/longitude (absent from the 9-field fixture → degrade to the FROZEN neutral
    # distance). The km column is float-coerced BEFORE np.log1p (object-dtype-crash defense).
    lat = df["latitude"] if "latitude" in df.columns else None
    lon = df["longitude"] if "longitude" in df.columns else None
    km, log_km, inv = center_distance_features(lat, lon, n)
    df["center_distance_km"] = km.to_numpy(dtype=float)
    df["log_center_distance"] = log_km.to_numpy(dtype=float)
    df["center_distance_inv"] = inv.to_numpy(dtype=float)

    # Interactions (mirror v20 :980-1017, magnitude-free shapes only).
    df["size_prime_interaction"] = df["size_sqft"] * df["is_prime_postcode"]
    df["size_x_central"] = df["size_sqft"] * df["center_distance_inv"]
    df["house_size_interaction"] = df["is_house"] * df["size_sqft"]
    df["prestige_tier_x_size"] = df["prestige_tier"] * df["size_sqft"]

    # Sale-only qualifier encoding (FEATURE-SAFE — derives from the price_qualifier STRING
    # ONLY, never the numeric label; AMENDMENT 1). Missing qualifier → "" → 0.
    if "price_qualifier" in df.columns:
        df["price_qualifier_poa"] = df["price_qualifier"].apply(
            lambda q: 1 if str("" if q is None else q).upper().startswith("POA") else 0
        )
    else:
        df["price_qualifier_poa"] = 0

    feature_cols = list(FEATURE_COLUMNS)
    X = df[feature_cols].astype(float)
    return X, feature_cols


def _targets(rows: list[dict]) -> np.ndarray:
    return np.array([r["asking_price"] for r in rows], dtype=float)


def _district_freq_map(rows: list[dict]) -> tuple[dict[str, float], float]:
    """Baked district-frequency map + its NUMERIC default (== min(map)), computed over the
    FULL rows (a leak-free property-attribute distribution). Mirrors the rental
    gen_inference_stats _assert_default_shape invariant by VALUE: the default is min(map),
    and the map itself embeds NO nested 'default' key."""
    districts = [postcode_to_district(r.get("postcode")) for r in rows]
    s = pd.Series(districts)
    freq = s.value_counts(normalize=True)
    fmap = {str(k): float(v) for k, v in freq.items()}
    default = float(min(fmap.values())) if fmap else 0.0
    return fmap, default


def _postcode_area_freq_map(rows: list[dict]) -> tuple[dict[str, float], float]:
    """Baked postcode-area-frequency map + its NUMERIC default (== min(map)). Coarser
    geographic bucket than the district; same leak-free / _assert_default_shape discipline."""
    areas = [postcode_to_area(r.get("postcode")) for r in rows]
    s = pd.Series(areas)
    freq = s.value_counts(normalize=True)
    amap = {str(k): float(v) for k, v in freq.items()}
    default = float(min(amap.values())) if amap else 0.0
    return amap, default


def train(rows: list[dict], seed: int = 42, test_size: float = 0.25) -> dict:
    """Train the baseline XGBoost sale model on log1p(asking_price).

    Drops rows whose asking_price or sale-ppsf is implausible (sale-magnitude data
    quality, not a feature). Returns {model, feature_cols, metrics{r2, mae, median_ape}}.
    """
    # Data-quality filter (sale magnitude / ppsf) — NOT leakage: it removes junk rows
    # before training, identical in spirit to the rental PPSF/price bounds in load_and_clean.
    clean: list[dict] = []
    for r in rows:
        price = r.get("asking_price")
        if not is_plausible_sale_price(price):
            continue
        size = r.get("size_sqft")
        if size:
            ppsf = float(price) / float(size)
            if not is_plausible_sale_ppsf(ppsf):
                continue
        clean.append(r)

    X, feature_cols = build_features(clean)
    y = np.log1p(_targets(clean))

    # MONOTONE size constraint (mirror the rental non-monotonicity DEFECT-CLASS fix; rental
    # has none, sale ADDS one). Aligned to feature_cols ORDER so it stays correct as the
    # column set grows.
    #
    # XGBoost's monotone_constraints only constrain the response w.r.t. the LISTED columns.
    # Constraining size_sqft/log_sqft alone is INSUFFICIENT here because price also rises
    # through several UNCONSTRAINED size-correlated columns (sqrt_sqft, size_per_bed,
    # size_squared, size_*interaction), and a tree can learn a local dip via one of those —
    # observed as a 700->800 sqft dip on this fixture (gate G4a / test_inc3_monotone_size).
    # To make price genuinely non-decreasing in size at fixed other inputs, we +1-constrain
    # EVERY feature that is itself a monotone-increasing function of size_sqft (each of these
    # only grows as size grows when all other inputs are held equal). This is the literal
    # realisation of "a monotone size constraint" — the broader tuple is what makes the
    # ordering hold, not just the two raw size columns.
    _SIZE_MONOTONE_COLS = (
        "size_sqft", "log_sqft", "sqrt_sqft", "size_per_bed", "size_squared",
        "size_prime_interaction", "size_x_central", "house_size_interaction",
        "prestige_tier_x_size",
    )
    mono = tuple(1 if c in _SIZE_MONOTONE_COLS else 0 for c in feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.0,
        monotone_constraints=mono,
        random_state=seed,
        n_jobs=1,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    actual = np.expm1(y_test)
    r2 = float(r2_score(y_test, pred_log))
    mae = float(mean_absolute_error(actual, pred))
    median_ape = float(np.median(np.abs(pred - actual) / actual))

    # Baked freq maps (computed on the FULL cleaned rows — a stable property-attribute
    # distribution, NOT the train split — so inference reproduces the training distribution).
    # Leak-free: it is the frequency of a categorical attribute, never the label.
    freq_map, freq_default = _district_freq_map(clean)
    area_freq_map, area_freq_default = _postcode_area_freq_map(clean)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": {"r2": r2, "mae": mae, "median_ape": median_ape},
        "n_train": len(X_train),
        "n_test": len(X_test),
        # ── NEW (additive — existing keys unchanged so the baseline tests still pass):
        "freq_map": freq_map,
        "freq_default": freq_default,
        "area_freq_map": area_freq_map,
        "area_freq_default": area_freq_default,
        "seed": seed,
    }


def predict(model, feature_cols: list[str], rows: list[dict]) -> np.ndarray:
    """Predict fair SALE value (£) for a list of property dicts. Handles the log1p/expm1
    round-trip and guarantees finite output for postcode-less / sparse rows."""
    X, built_cols = build_features(rows)
    # Align to the trained feature order (zero-fill any the trainer had but a sparse
    # inference row lacks — the rental serving pattern).
    X = X.reindex(columns=feature_cols, fill_value=0.0).astype(float)
    pred_log = model.predict(X)
    return np.expm1(pred_log)


def predict_one(model, feature_cols: list[str], **kwargs) -> float:
    """Single-property convenience predictor (mirrors canonical_predict.predict_one)."""
    return float(predict(model, feature_cols, [kwargs])[0])


def save_model(model, feature_cols: list[str],
               model_path: Path | str = DEFAULT_MODEL_PATH,
               features_path: Path | str = DEFAULT_FEATURES_PATH) -> None:
    """Serialize the sale model + feature order to the SEPARATE sale artifact paths."""
    model_path = Path(model_path)
    features_path = Path(features_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(features_path, "wb") as f:
        pickle.dump(list(feature_cols), f)


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH,
               features_path: Path | str = DEFAULT_FEATURES_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols
