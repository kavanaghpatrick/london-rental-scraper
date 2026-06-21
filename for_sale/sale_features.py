"""for_sale.sale_features — PURE, tenure-agnostic feature primitives for the for-sale model.

These are the load-bearing helpers that grew large enough to warrant their own module:
the EXTENDED property-type classifiers (penthouse / maisonette / terraced / studio), the
sale-side CITY-CENTRE distance constants, and the coordinate/distance feature helpers with
the object-dtype-crash defense (coerce to float BEFORE np.log1p).

ISOLATION CONTRACT
------------------
This module imports NOTHING from the rental chain (rental_price_models_v20 /
canonical_predict / retrain_canonical / the rental sidecar generators). Every tenure-
agnostic pattern below is RE-IMPLEMENTED BY VALUE, mirroring the rental v20 shapes so a
rental retrain can never perturb the sale model and vice-versa. The only legal cross-
package import in the for-sale vertical lives in sale_data.py (the fingerprint primitive);
this file reaches for stdlib + numpy/pandas only.

NEUTRAL-FALLBACK DISCIPLINE
---------------------------
The committed CI fixture carries only 9 fields (no latitude/longitude). Every coordinate
feature MUST degrade to a FROZEN neutral constant when its source field is absent — NOT to
0 (a city-centre distance of 0 km is out-of-distribution: the training distribution has no
rows at the centroid). The neutral constant is the sale-frame median-ish distance, mirroring
the rental object-dtype-crash defense (frozen training-median constants, never CITY_CENTER).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Sale-side CITY-CENTRE + neutral distance (mirror the rental frozen-constant defense) ──
# DEFAULT_CENTER_DISTANCE_KM is the NEUTRAL fill for coordless rows. It is deliberately NOT
# 0.0: a city-centre distance of exactly 0 is out-of-distribution (no real listing sits on
# the centroid), so filling 0 would score every coordless row maximally-central. We freeze a
# representative non-zero km instead (the rental side froze its training-median distance for
# the identical reason). SALE_CITY_CENTER is the reference centroid used to compute the
# Haversine distance for rows that DO carry coordinates.
DEFAULT_CENTER_DISTANCE_KM = 3.3892584524370477  # neutral, NOT 0 (centroid is OOD)
SALE_CITY_CENTER = (51.5074, -0.1278)  # (lat, lon) — central London reference point

_EARTH_RADIUS_KM = 6371.0088


# ── Extended property-type classifiers (mirror rental v20 :822-846, RE-IMPLEMENTED) ──────
# Source field: row["property_type"] string (the fixture vocabulary is Flat / Town House /
# House / Apartment / Maisonette / Penthouse — Finding 3). Each classifier is a pure
# string-match returning a 0/1 int. A plain "Flat" / "Apartment" sets NONE of the four.
def _classify_type_extended(property_type: str | None) -> tuple[int, int, int, int]:
    """Return (is_penthouse, is_maisonette, is_terraced, is_studio) from the raw type string.

    - is_penthouse : the type mentions a penthouse.
    - is_maisonette: the type mentions a maisonette (or duplex, its close cousin).
    - is_terraced  : the type is a terraced house family member (terraced / town house / end
                     of terrace), i.e. the house-side analogue the rental model flags.
    - is_studio    : the type mentions a studio (a string match — keeps a coordless / size-
                     less row classifiable; the rental side also uses a size heuristic, but
                     the sale fixture labels studios explicitly).
    A plain Flat / Apartment matches none of the four. Mirrors rental v20 by value.
    """
    pt = (property_type or "").lower()
    is_penthouse = int("penthouse" in pt)
    is_maisonette = int("maisonette" in pt or "duplex" in pt)
    is_terraced = int(
        "terraced" in pt or "town house" in pt or "townhouse" in pt or "end of terrace" in pt
    )
    is_studio = int("studio" in pt)
    return is_penthouse, is_maisonette, is_terraced, is_studio


def _to_float_series(values, length: int) -> pd.Series:
    """Coerce an arbitrary column (possibly object dtype / missing) to a float Series.

    Returns an all-NaN float Series of the given length when `values` is None/absent. This is
    the guard that prevents the np.log1p object-dtype crash class: callers coerce to float
    BEFORE any np.log1p / sqrt is applied (Finding 1/2)."""
    if values is None:
        return pd.Series([np.nan] * length, dtype=float)
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").astype(float)


def _haversine_km(lat, lon, center=SALE_CITY_CENTER) -> pd.Series:
    """Vectorized great-circle distance (km) from each (lat, lon) to `center`.

    lat/lon are coerced to float first (object-dtype-safe). Rows whose coordinate is NaN
    yield NaN here — the caller fills those with DEFAULT_CENTER_DISTANCE_KM (the neutral
    coordless fallback). Mirrors the rental distance feature shape by value."""
    clat, clon = center
    lat_f = pd.to_numeric(pd.Series(list(lat)), errors="coerce").astype(float)
    lon_f = pd.to_numeric(pd.Series(list(lon)), errors="coerce").astype(float)

    lat1 = np.radians(lat_f.to_numpy())
    lon1 = np.radians(lon_f.to_numpy())
    lat2 = np.radians(float(clat))
    lon2 = np.radians(float(clon))

    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    km = 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return pd.Series(km, dtype=float)


def center_distance_features(
    latitude, longitude, length: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build the (center_distance_km, log_center_distance, center_distance_inv) triple.

    Coordless rows (missing or NaN lat/long) degrade to the FROZEN neutral constant
    DEFAULT_CENTER_DISTANCE_KM (NOT 0). The km column is float-coerced BEFORE np.log1p so a
    sparse / object-dtype input can never trigger the np.log1p object-dtype crash (the bug
    class the rental side hit). All three returned Series are finite floats of `length`.

    - center_distance_km  : great-circle km to SALE_CITY_CENTER, neutral-filled if coordless.
    - log_center_distance : np.log1p of the float-coerced km (monotone compression).
    - center_distance_inv : 1 / (1 + km) — a magnitude-free "centrality" shape (1.0 at the
                            centroid, →0 far out).
    """
    if latitude is None or longitude is None:
        km = pd.Series([DEFAULT_CENTER_DISTANCE_KM] * length, dtype=float)
    else:
        km = _haversine_km(latitude, longitude)
        # Neutral-fill any coordless / unparseable row (NaN) — NOT 0 (centroid is OOD).
        km = km.fillna(DEFAULT_CENTER_DISTANCE_KM).astype(float)

    # Coerce to float BEFORE log1p — the object-dtype-crash defense (Finding 1/2).
    km = pd.to_numeric(km, errors="coerce").fillna(DEFAULT_CENTER_DISTANCE_KM).astype(float)
    log_km = np.log1p(km.to_numpy(dtype=float))
    inv = 1.0 / (1.0 + km.to_numpy(dtype=float))
    return (
        km.reset_index(drop=True),
        pd.Series(log_km, dtype=float),
        pd.Series(inv, dtype=float),
    )
