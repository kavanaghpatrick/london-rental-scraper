#!/usr/bin/env python3
"""
sync_sale_js_freq_maps.py — re-bake the SALE Chrome/serving predictor's baked frequency
maps from output/sale_model_inference.json, then re-vendor the dashboard copy.

WHY (sale-vertical retrain-regen drift — the analogue of task #14 for the rental script):
  The SALE XGBoost JS predictor (chrome-extension/sale_xgboost.js) carries BAKED
  DISTRICT_FREQ / POSTCODE_AREA_FREQ maps (+ their *_DEFAULT scalars) that MUST equal the
  Python training frequencies in output/sale_model_inference.json — the same maps the sale
  model injects at single-row inference. When a REAL retrain regenerates inference.json
  (new n_train -> new freqs / new defaults) but the JS map is NOT re-baked, JS<->Python
  parity silently drifts and chrome-extension/sale_fixture_diff.mjs fails. Run this as part
  of EVERY sale retrain (.github/workflows/for-sale-scrape.yml) so the JS map, the served
  model, and the golden fixture move together atomically.

SALE SHAPE (LOAD-BEARING — DIFFERENT from the rental script):
  The sale JS uses SEPARATE constants, NOT the rental "folded 'default' key" form, and there
  is NO POSTCODE_FREQ (district-level + area-level only):
    DISTRICT_FREQ: { ... },            <- inference["district_freq"]
    DISTRICT_FREQ_DEFAULT: <num>,      <- inference["district_freq_default"]
    POSTCODE_AREA_FREQ: { ... },       <- inference["postcode_area_freq"]
    POSTCODE_AREA_FREQ_DEFAULT: <num>, <- inference["postcode_area_freq_default"]
  The map literals are rendered 3 entries per line, matching the committed file byte-for-byte
  so this script is a strict NO-OP on a self-consistent tree (the synthetic committed maps
  already match inference.json — it only does work once a real retrain changes the maps).

Idempotent: a no-op (exit 0) when the JS maps already match inference.json AND the vendored
copy matches the extension file.

Usage:
  python3 scripts/sync_sale_js_freq_maps.py
  python3 scripts/sync_sale_js_freq_maps.py --check   # verify-only, non-zero if drifted
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INFERENCE = REPO / "output" / "sale_model_inference.json"
SALE_JS = REPO / "chrome-extension" / "sale_xgboost.js"
VENDORED = REPO / "dashboard" / "src" / "app" / "api" / "predict-sale" / "sale_xgboost.predictor.js"

# Inference keys the sale JS consumes (separate-constant shape — NO folded default).
_REQUIRED_KEYS = (
    "district_freq",
    "district_freq_default",
    "postcode_area_freq",
    "postcode_area_freq_default",
)

# How many map entries per rendered line. Matches the committed SALE JS exactly so a
# self-consistent tree round-trips byte-for-byte (no spurious reformat-only diff).
_ENTRIES_PER_LINE = 3


def _js_obj(values: dict) -> str:
    """
    Render a JS object literal for a freq map in the SALE shape: quoted keys, repr floats,
    a trailing comma after the LAST entry, _ENTRIES_PER_LINE entries per line, indented to
    match the surrounding SaleXGBFeatures block. NO folded 'default' key (the sale JS keeps
    the default in a separate *_DEFAULT constant).
    """
    items = [f"'{k}': {float(v)!r}" for k, v in values.items()]
    lines = []
    for i in range(0, len(items), _ENTRIES_PER_LINE):
        chunk = items[i:i + _ENTRIES_PER_LINE]
        lines.append("    " + ", ".join(chunk) + ",")
    return "{\n" + "\n".join(lines) + "\n  }"


def _build(src: str, inf: dict) -> str:
    """
    Apply the four targeted, count=1 substitutions for the four SALE labels. Each is anchored
    on the exact 2-space-indented label so a stray match elsewhere is impossible.
    """
    out = re.sub(
        r"(  DISTRICT_FREQ:\s*)\{.*?\},",
        lambda m: m.group(1) + _js_obj(inf["district_freq"]) + ",",
        src, count=1, flags=re.S,
    )
    out = re.sub(
        r"(  DISTRICT_FREQ_DEFAULT:\s*)[0-9.eE+-]+,",
        lambda m: m.group(1) + repr(float(inf["district_freq_default"])) + ",",
        out, count=1,
    )
    out = re.sub(
        r"(  POSTCODE_AREA_FREQ:\s*)\{.*?\},",
        lambda m: m.group(1) + _js_obj(inf["postcode_area_freq"]) + ",",
        out, count=1, flags=re.S,
    )
    out = re.sub(
        r"(  POSTCODE_AREA_FREQ_DEFAULT:\s*)[0-9.eE+-]+,",
        lambda m: m.group(1) + repr(float(inf["postcode_area_freq_default"])) + ",",
        out, count=1,
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true",
        help="verify-only: exit non-zero if the SALE JS maps have drifted from inference.json",
    )
    args = ap.parse_args(argv)

    for p in (INFERENCE, SALE_JS, VENDORED):
        if not p.exists():
            print(f"[sync_sale_js_freq_maps] ERROR: missing {p}", file=sys.stderr)
            return 2

    inf = json.loads(INFERENCE.read_text())
    for k in _REQUIRED_KEYS:
        if k not in inf:
            print(f"[sync_sale_js_freq_maps] ERROR: inference.json missing '{k}'", file=sys.stderr)
            return 2

    src = SALE_JS.read_text()
    rebuilt = _build(src, inf)

    if rebuilt == src:
        print("[sync_sale_js_freq_maps] sale_xgboost.js DISTRICT_FREQ/POSTCODE_AREA_FREQ "
              "already in sync with sale_model_inference.json")
    else:
        if args.check:
            print("[sync_sale_js_freq_maps] DRIFT: sale_xgboost.js freq maps != "
                  "sale_model_inference.json. Run without --check to re-bake "
                  "(a real retrain must regen this).", file=sys.stderr)
            return 1
        # Guard against the regex silently matching nothing (e.g. a renamed label) and
        # writing an unchanged-but-stale file: confirm all four labels survive the rewrite.
        for label in ("DISTRICT_FREQ:", "DISTRICT_FREQ_DEFAULT:",
                      "POSTCODE_AREA_FREQ:", "POSTCODE_AREA_FREQ_DEFAULT:"):
            if label not in rebuilt:
                print(f"[sync_sale_js_freq_maps] ERROR: could not locate {label} literal "
                      "to rewrite", file=sys.stderr)
                return 2
        SALE_JS.write_text(rebuilt)
        print(f"[sync_sale_js_freq_maps] re-baked sale_xgboost.js freq maps from "
              f"sale_model_inference.json (district_freq: {len(inf['district_freq'])} entries, "
              f"default={inf['district_freq_default']}; postcode_area_freq: "
              f"{len(inf['postcode_area_freq'])} entries, default={inf['postcode_area_freq_default']})")

    # Re-vendor byte-identical (the dashboard /api/predict-sale route imports the SAME source,
    # so client + server feature builders can never drift). Self-heal pre-existing vendored
    # drift even on a no-op of the maps.
    if VENDORED.read_bytes() != SALE_JS.read_bytes():
        if args.check:
            print("[sync_sale_js_freq_maps] DRIFT: vendored sale predictor != "
                  "chrome-extension/sale_xgboost.js", file=sys.stderr)
            return 1
        shutil.copy(SALE_JS, VENDORED)
        # Show a repo-relative path when possible; fall back to the name (tests redirect
        # VENDORED to a tmp dir outside the repo, so relative_to() would raise there).
        try:
            shown = VENDORED.relative_to(REPO)
        except ValueError:
            shown = VENDORED.name
        print(f"[sync_sale_js_freq_maps] re-vendored {shown} (byte-identical)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
