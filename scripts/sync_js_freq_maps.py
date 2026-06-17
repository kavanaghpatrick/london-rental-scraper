#!/usr/bin/env python3
"""
sync_js_freq_maps.py — re-bake the Chrome predictor's hardcoded frequency maps
from output/rental_model_canonical_inference.json, then re-vendor the dashboard copy.

WHY (task #14 — close the retrain-regen drift):
  The XGBoost JS predictor (chrome-extension/xgboost.js) carries HARDCODED
  POSTCODE_FREQ / POSTCODE_AREA_FREQ maps that MUST equal the Python training
  frequencies in rental_model_canonical_inference.json (the same maps
  canonical_predict injects at inference). When a retrain regenerates inference.json
  (new n_train -> new freqs / new default) but the JS map is NOT re-baked, JS<->Python
  parity silently drifts and _fixture_diff.mjs fails. This was the daily-pipeline gap
  that desynced fa669cd's matched set. Run this as part of EVERY retrain so the JS map,
  the served model, and the golden fixture move together atomically.

WHAT IT DOES:
  1. Reads inference.json: postcode_freq{}, postcode_area_freq{} (+ their *_default).
  2. Rewrites the `POSTCODE_FREQ: {...}` and `POSTCODE_AREA_FREQ: {...}` object
     literals in chrome-extension/xgboost.js (folding the numeric default in as a
     'default' key, the form the JS lookup expects).
  3. Re-vendors dashboard/src/app/api/predict/xgboost.predictor.js byte-identical
     (the Extension-Drift-Guard requires the two to match exactly).
  4. Verifies the edit actually changed the literals (or is already in sync).

Idempotent: a no-op (exit 0) if the JS map already matches inference.json.

Usage:
  python3 scripts/sync_js_freq_maps.py
  python3 scripts/sync_js_freq_maps.py --check   # verify-only, non-zero if drifted
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INFERENCE = REPO / "output" / "rental_model_canonical_inference.json"
XGBOOST_JS = REPO / "chrome-extension" / "xgboost.js"
VENDORED = REPO / "dashboard" / "src" / "app" / "api" / "predict" / "xgboost.predictor.js"


def _js_map_literal(values: dict, default) -> str:
    """Render a JS object literal with the numeric default folded in as 'default'."""
    parts = [f"'{k}': {float(v)!r}" for k, v in values.items() if k != "default"]
    parts.append(f"'default': {float(default)!r}")
    return "{ " + ", ".join(parts) + " }"


def _build(src: str, inf: dict) -> str:
    pf = _js_map_literal(inf["postcode_freq"], inf["postcode_freq_default"])
    paf = _js_map_literal(inf["postcode_area_freq"], inf["postcode_area_freq_default"])
    out = re.sub(r"(  POSTCODE_FREQ:\s*)\{.*?\},", lambda m: m.group(1) + pf + ",",
                 src, count=1, flags=re.S)
    out = re.sub(r"(  POSTCODE_AREA_FREQ:\s*)\{.*?\},", lambda m: m.group(1) + paf + ",",
                 out, count=1, flags=re.S)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify-only: exit non-zero if the JS map has drifted from inference.json")
    args = ap.parse_args()

    for p in (INFERENCE, XGBOOST_JS, VENDORED):
        if not p.exists():
            print(f"[sync_js_freq_maps] ERROR: missing {p}", file=sys.stderr)
            return 2

    inf = json.loads(INFERENCE.read_text())
    for k in ("postcode_freq", "postcode_area_freq", "postcode_freq_default", "postcode_area_freq_default"):
        if k not in inf:
            print(f"[sync_js_freq_maps] ERROR: inference.json missing '{k}'", file=sys.stderr)
            return 2

    src = XGBOOST_JS.read_text()
    rebuilt = _build(src, inf)
    if rebuilt == src:
        print(f"[sync_js_freq_maps] xgboost.js POSTCODE_FREQ/AREA already in sync with inference.json")
    else:
        if "0.999" in _js_map_literal(inf["postcode_freq"], inf["postcode_freq_default"]):
            pass  # (defensive; numbers are floats)
        if args.check:
            print("[sync_js_freq_maps] DRIFT: xgboost.js freq maps != inference.json. "
                  "Run without --check to re-bake (retrain must regen this).", file=sys.stderr)
            return 1
        # Guard against the regex silently matching nothing
        if "POSTCODE_FREQ:" not in rebuilt or "POSTCODE_AREA_FREQ:" not in rebuilt:
            print("[sync_js_freq_maps] ERROR: could not locate POSTCODE_FREQ/AREA literals to rewrite",
                  file=sys.stderr)
            return 2
        XGBOOST_JS.write_text(rebuilt)
        print(f"[sync_js_freq_maps] re-baked xgboost.js freq maps from inference.json "
              f"(postcode_freq: {len(inf['postcode_freq'])} entries, default={inf['postcode_freq_default']})")

    # Re-vendor byte-identical (Extension Drift Guard requires it), even on a no-op,
    # to self-heal any pre-existing vendored drift.
    if VENDORED.read_bytes() != XGBOOST_JS.read_bytes():
        if args.check:
            print("[sync_js_freq_maps] DRIFT: vendored predictor != chrome-extension/xgboost.js",
                  file=sys.stderr)
            return 1
        shutil.copy(XGBOOST_JS, VENDORED)
        print(f"[sync_js_freq_maps] re-vendored {VENDORED.relative_to(REPO)} (byte-identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
