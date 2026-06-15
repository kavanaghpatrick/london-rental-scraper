#!/usr/bin/env python3
"""
generate_predictions.py — Build the predictions cache from the CANONICAL model.

Replaces the inline, v15-era feature dict that used to live in
`.github/workflows/generate-predictions.yml`. That inline builder hand-coded ~50
features and silently drifted from the served model (bug: predictions computed with
a different feature schema than the model was trained on). This script instead:

  1. Loads the canonical model + feature order:
        output/rental_model_canonical.pkl
        output/rental_model_canonical_features.pkl   (135-col contract)
  2. Reads ACTIVE listings from the SAME store the dashboard serves:
        - Postgres (POSTGRES_URL) by default — this is what the scrape writes in CI;
        - falls back to local SQLite (output/rentals.db) for offline/dev runs.
  3. Predicts via canonical_predict.predict() — the SINGLE linchpin that builds
     features + loads the canonical model + inverts log1p. No feature math here;
     the prediction-time schema is byte-for-byte the training schema.
  4. Writes `chrome-extension/api/predictions.json`, keyed `source:property_id`,
     values {fv, lo, hi, pct, sq}.

Single source of truth for feature construction = canonical_predict.py
(which itself resolves the canonical FE module from retrain_canonical.py).
Run:
    python3 scripts/generate_predictions.py                      # Postgres if POSTGRES_URL set, else SQLite
    python3 scripts/generate_predictions.py --sqlite output/rentals.db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Import the modeler's blessed canonical module (the single source of truth for
# loading the model + building features + predicting). Lead's ruling: import, not
# reimplement. canonical_predict itself stays version-agnostic via retrain_canonical.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import canonical_predict as cp  # noqa: E402

OUT_PATH = PROJECT_ROOT / "chrome-extension" / "api" / "predictions.json"

# Raw listing columns canonical_predict.build_features() needs to engineer features.
# Pulled from whichever store we read so the canonical FE pipeline has everything.
SELECT_COLUMNS = [
    "source", "property_id", "url", "price_pcm", "bedrooms", "bathrooms",
    "size_sqft", "postcode", "area", "property_type", "address",
    "latitude", "longitude", "features", "description",
    "property_type_std", "let_type", "postcode_normalized", "postcode_inferred",
    "agent_brand", "floor_count", "has_roof_terrace", "has_basement", "has_ground",
    "has_first_floor", "has_second_floor", "has_third_floor", "has_fourth_plus",
]


def load_listings_postgres(url: str) -> pd.DataFrame:
    import psycopg2  # imported lazily so SQLite-only runs don't need it

    cols = ", ".join(SELECT_COLUMNS)
    query = (
        f"SELECT {cols} FROM listings "
        "WHERE is_active = 1 AND price_pcm > 500 AND size_sqft > 0 "
        "AND bedrooms IS NOT NULL"
    )
    conn = psycopg2.connect(url)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    print(f"Loaded {len(df)} active listings from Postgres")
    return df


def load_listings_sqlite(path: str) -> pd.DataFrame:
    import sqlite3

    # Defensive: only select columns that exist in this SQLite schema.
    conn = sqlite3.connect(path)
    try:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(listings)")}
        cols = [c for c in SELECT_COLUMNS if c in existing]
        query = (
            f"SELECT {', '.join(cols)} FROM listings "
            "WHERE is_active = 1 AND price_pcm > 500 AND size_sqft > 0 "
            "AND bedrooms IS NOT NULL"
        )
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    print(f"Loaded {len(df)} active listings from SQLite ({path})")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default=None, help="Force SQLite source (path to rentals.db)")
    ap.add_argument("--postgres-url", default=None, help="Override POSTGRES_URL")
    args = ap.parse_args()

    # Source selection: explicit --sqlite wins; else Postgres if URL present; else local SQLite.
    pg_url = args.postgres_url or os.environ.get("POSTGRES_URL")
    if args.sqlite:
        df = load_listings_sqlite(args.sqlite)
    elif pg_url:
        df = load_listings_postgres(pg_url)
    else:
        default_db = PROJECT_ROOT / "output" / "rentals.db"
        df = load_listings_sqlite(str(default_db))

    if df.empty:
        # Write an empty cache rather than crash — matches "no active listings" state.
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text("{}")
        print("WARNING: 0 active listings — wrote empty predictions.json {}")
        return 0

    # Keep identity columns to key the output.
    ids = (df["source"].astype(str) + ":" + df["property_id"].astype(str)).tolist()
    asking = df["price_pcm"].tolist()
    raw_sqft = df["size_sqft"].tolist()

    # Delegate ALL feature building + prediction to the modeler's blessed module
    # (canonical_predict.predict): loads the canonical model, builds features via the
    # canonical pipeline, aligns column order, inverts log1p. No inline feature math.
    print(f"Predicting {len(df)} listings via canonical_predict ({cp.CANONICAL_VERSION})...")
    fair_values = cp.predict(df)

    predictions = {}
    for key, fv, ask, sq in zip(ids, fair_values, asking, raw_sqft):
        fv = int(fv)
        if fv <= 0:
            continue
        pct = round((ask / fv - 1) * 100, 1) if fv > 0 else 0
        predictions[key] = {
            "fv": fv,
            "lo": int(fv * 0.79),
            "hi": int(fv * 1.21),
            "pct": pct,
            "sq": int(sq) if sq else None,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(predictions, f, separators=(",", ":"))
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Generated {len(predictions)} predictions -> {OUT_PATH} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
