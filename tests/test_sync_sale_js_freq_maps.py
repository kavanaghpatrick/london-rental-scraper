"""
TDD guard tests for scripts/sync_sale_js_freq_maps.py (spec §3.1, Writer3).

The SALE JS predictor (chrome-extension/sale_xgboost.js + its byte-identical vendored
copy dashboard/src/app/api/predict-sale/sale_xgboost.predictor.js) carries BAKED
frequency maps that MUST equal the Python training frequencies in
output/sale_model_inference.json. After a REAL retrain regenerates inference.json,
sync_sale_js_freq_maps.py re-bakes those maps into the JS, re-vendors the dashboard
copy byte-identically, and exposes a --check mode that exits non-zero on drift.

CRITICAL (sale differs from rental): the sale JS uses SEPARATE constants
  DISTRICT_FREQ / DISTRICT_FREQ_DEFAULT / POSTCODE_AREA_FREQ / POSTCODE_AREA_FREQ_DEFAULT
and the inference keys district_freq / district_freq_default / postcode_area_freq /
postcode_area_freq_default. There is NO folded 'default' key and NO POSTCODE_FREQ.

These tests operate on TMP COPIES only (monkeypatching the module path globals); they
NEVER touch the tracked chrome-extension/sale_xgboost.js or the vendored predictor —
those are rewritten only at GHA runtime by a real retrain.
"""
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

# Load the script by file path. A top-level `import scripts.sync_sale_js_freq_maps`
# is unreliable here because an UNRELATED `scripts` package is installed in
# site-packages and shadows the repo's scripts/ directory. Loading by path targets
# the repo file unambiguously.
#
# RED expectation (before implementation): the file scripts/sync_sale_js_freq_maps.py
# does not exist, so spec_from_file_location returns a loader that raises
# FileNotFoundError on exec -> the whole module errors at collection and every
# T-C* test FAILS. After the script lands, the import resolves and the tests run.
_SCRIPT_PATH = _REPO / "scripts" / "sync_sale_js_freq_maps.py"
_spec = importlib.util.spec_from_file_location("sync_sale_js_freq_maps", _SCRIPT_PATH)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


# --- Real tracked sources (read-only; never mutated by the tests) ---
_REAL_INFERENCE = _REPO / "output" / "sale_model_inference.json"
_REAL_SALE_JS = _REPO / "chrome-extension" / "sale_xgboost.js"
_REAL_VENDORED = _REPO / "dashboard" / "src" / "app" / "api" / "predict-sale" / "sale_xgboost.predictor.js"


@pytest.fixture
def tmp_tree(tmp_path, monkeypatch):
    """
    Build a throwaway copy of the three real files in tmp_path and point the module's
    path globals at them. The tracked files are NEVER written by these tests.
    """
    inf = tmp_path / "sale_model_inference.json"
    js = tmp_path / "sale_xgboost.js"
    vendored = tmp_path / "sale_xgboost.predictor.js"
    inf.write_bytes(_REAL_INFERENCE.read_bytes())
    js.write_bytes(_REAL_SALE_JS.read_bytes())
    vendored.write_bytes(_REAL_VENDORED.read_bytes())

    monkeypatch.setattr(sync, "INFERENCE", inf)
    monkeypatch.setattr(sync, "SALE_JS", js)
    monkeypatch.setattr(sync, "VENDORED", vendored)
    return {"inf": inf, "js": js, "vendored": vendored}


def _set_inference_district(inf_path: Path, district: str, value: float):
    """Rewrite one district_freq entry to a distinctive value (forces drift)."""
    data = json.loads(inf_path.read_text())
    data["district_freq"][district] = value
    inf_path.write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------------
# T-C1 — a re-bake writes the inference values into the JS, and re-vendors identically
# --------------------------------------------------------------------------------
def test_rebake_writes_inference_values_into_js(tmp_tree):
    distinctive = 0.42424242424242   # NOT any value already in the synthetic maps
    _set_inference_district(tmp_tree["inf"], "SW3", distinctive)

    rc = sync.main([])
    assert rc == 0

    js_text = tmp_tree["js"].read_text()
    # repr(float(...)) is exactly how the renderer emits the number into the JS literal
    assert repr(distinctive) in js_text, "re-baked SW3 value not found in DISTRICT_FREQ"
    assert f"'SW3': {distinctive!r}" in js_text

    # vendored copy must be byte-identical to the freshly-baked extension file
    assert tmp_tree["vendored"].read_bytes() == tmp_tree["js"].read_bytes()


# --------------------------------------------------------------------------------
# T-C2 — --check detects drift (stale JS vs inference) and exits 1
# --------------------------------------------------------------------------------
def test_check_mode_detects_drift(tmp_tree):
    # Make inference disagree with the (still-synthetic) JS.
    _set_inference_district(tmp_tree["inf"], "SW3", 0.42424242424242)

    rc = sync.main(["--check"])
    assert rc == 1, "--check must return 1 when the JS map has drifted from inference.json"

    # --check must NOT have written anything: the JS still holds the OLD value.
    assert "0.42424242424242" not in tmp_tree["js"].read_text()


# --------------------------------------------------------------------------------
# T-C3 — after a re-bake, --check is a clean no-op (exit 0)
# --------------------------------------------------------------------------------
def test_check_mode_clean_is_zero(tmp_tree):
    _set_inference_district(tmp_tree["inf"], "SW3", 0.42424242424242)

    assert sync.main([]) == 0          # re-bake
    assert sync.main(["--check"]) == 0  # now in sync -> no-op

    # Idempotence: a second re-bake produces byte-stable output.
    before = tmp_tree["js"].read_bytes()
    assert sync.main([]) == 0
    assert tmp_tree["js"].read_bytes() == before


# --------------------------------------------------------------------------------
# T-C4 — regression lock: the REAL committed tree is self-consistent (JS==vendored
#        and JS already matches inference) so --check is a no-op (exit 0) on it.
#        This proves the new script does NOT perturb the current matched set (G4).
# --------------------------------------------------------------------------------
def test_real_files_in_sync():
    # Operate on the REAL module globals (no monkeypatch) but READ-ONLY: --check
    # never writes. If the committed tree were drifted this would (correctly) fail.
    assert _REAL_VENDORED.read_bytes() == _REAL_SALE_JS.read_bytes(), \
        "vendored predictor must be byte-identical to chrome-extension/sale_xgboost.js"

    rc = sync.main(["--check"])
    assert rc == 0, (
        "sync_sale_js_freq_maps.py --check is not a no-op on the committed tree; "
        "the baked SALE JS maps drifted from output/sale_model_inference.json"
    )

    # Belt: the script must NOT have mutated the tracked files (--check is read-only).
    assert _REAL_SALE_JS.read_bytes() == (_REPO / "chrome-extension" / "sale_xgboost.js").read_bytes()


# --------------------------------------------------------------------------------
# Extra: missing inference keys / missing files -> exit 2 (defensive), and the
# renderer produces the SALE separate-constant shape (NO folded 'default', NO POSTCODE_FREQ).
# --------------------------------------------------------------------------------
def test_missing_file_returns_2(tmp_tree, monkeypatch):
    tmp_tree["inf"].unlink()
    assert sync.main(["--check"]) == 2


def test_missing_inference_key_returns_2(tmp_tree):
    data = json.loads(tmp_tree["inf"].read_text())
    del data["district_freq_default"]
    tmp_tree["inf"].write_text(json.dumps(data, indent=2))
    assert sync.main(["--check"]) == 2


def test_renderer_is_sale_shape_not_rental(tmp_tree):
    # Force a re-bake so the renderer's output is materialized in the JS.
    _set_inference_district(tmp_tree["inf"], "SW3", 0.42424242424242)
    assert sync.main([]) == 0
    js_text = tmp_tree["js"].read_text()
    # The SALE JS must keep its four separate labels intact.
    for label in ("DISTRICT_FREQ:", "DISTRICT_FREQ_DEFAULT:", "POSTCODE_AREA_FREQ:",
                  "POSTCODE_AREA_FREQ_DEFAULT:"):
        assert label in js_text, f"{label} must remain in the SALE JS after re-bake"
    # And it must NOT grow a rental-style folded 'default' key or a POSTCODE_FREQ map.
    assert "POSTCODE_FREQ:" not in js_text, "sale JS must not contain a rental POSTCODE_FREQ map"
    # The DISTRICT_FREQ block must not fold a numeric default in as a 'default' key.
    m = re.search(r"  DISTRICT_FREQ:\s*\{(.*?)\},", js_text, flags=re.S)
    assert m is not None
    assert "'default'" not in m.group(1), "sale DISTRICT_FREQ must not fold a 'default' key"
