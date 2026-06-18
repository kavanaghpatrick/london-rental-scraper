"""
test_js_python_parity.py — HARD CI GATE: the JS extension/serving feature builder
(chrome-extension/xgboost.js buildFeatures + predict) must stay byte-equal to the
Python canonical model's golden fixture (task #47, architecture B).

This wraps chrome-extension/fixture_diff.mjs (the key-by-key drift guard) as a
pytest test so the parity check runs from the single `pytest` CI entrypoint and
FAILS the build on ANY feature or £ mismatch across the 7 golden samples.

Requires: node on PATH + output/feature_parity_golden.json + chrome-extension/.
Skips (does NOT fail) only if node or the fixture is absent — so a Python-only
environment can still run the rest of the suite; CI must have node.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DIFF = ROOT / 'chrome-extension' / 'fixture_diff.mjs'
FIXTURE = ROOT / 'output' / 'feature_parity_golden.json'
XGBOOST_JS = ROOT / 'chrome-extension' / 'xgboost.js'

_NODE = shutil.which('node')


@pytest.mark.parity
@pytest.mark.skipif(_NODE is None, reason="node not on PATH (CI must provide it)")
@pytest.mark.skipif(not DIFF.exists(), reason="fixture_diff.mjs missing")
@pytest.mark.skipif(not FIXTURE.exists(),
                    reason="golden fixture missing — run gen_feature_parity_golden.py")
@pytest.mark.skipif(not XGBOOST_JS.exists(), reason="xgboost.js missing")
def test_js_python_golden_parity():
    """node fixture_diff.mjs must exit 0 (0 feature + 0 £ mismatch, all 7 samples)."""
    proc = subprocess.run(
        [_NODE, str(DIFF), str(FIXTURE)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    # The mjs prints a PASS/FAIL summary + per-sample lines; surface them on failure.
    assert proc.returncode == 0, (
        "JS↔Python parity FAILED — xgboost.js buildFeatures/predict drifted from the "
        "golden fixture. Output:\n" + proc.stdout[-4000:] + "\n" + proc.stderr[-2000:]
    )
    assert 'PASS:' in proc.stdout, f"unexpected parity output:\n{proc.stdout[-2000:]}"
