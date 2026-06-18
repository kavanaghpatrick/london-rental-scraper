"""T3 — RETRAIN VALIDATION GATES (TDD, test-FIRST; gates the SHIP of the retrained model).

This file encodes R3's ship gates as RUNNABLE pytest checks. They are written BEFORE
the human runs the retrain, and are PARAMETERIZED on the post-retrain artifacts so the
same file is the gate the human runs AFTER `python3 retrain_canonical.py --db output/rentals.db`.

WHAT MUST BE TRUE TO SHIP THE RETRAINED MODEL (each is one gate below):
  G1  PARITY 0/0 ............ node chrome-extension/fixture_diff.mjs reports
                              "0 feature mismatch(es), 0 £ mismatch(es)" against the
                              freshly-regenerated golden fixture + exported api/ artifacts.
  G2  FEATURE COUNT SANE .... the retrained feature list is still 150 (v20 schema unchanged);
                              the model's n_features in pkl == features pkl length == meta.
  G3  NO STRING/OBJECT FEATURE (LEAKAGE / SCHEMA GUARD) — none of the selected feature
                              columns are object/string dtype when re-engineered (a string
                              column slipping into X is both a train-time crash AND the classic
                              vector for an accidental target-leak / id-like feature). The
                              backfill adds DATA (size_sqft), never a feature, so the schema
                              must be byte-identical to the shipped 150.
  G4  R²/MAE NOT REGRESSED ... NEW 5-fold CV on the trained DB must be >= the BEFORE baseline
                              on the SAME DB minus a small band, and MAE must not blow out.
                              CRITICAL (R3): compare AFTER-rentals.db vs BEFORE-rentals.db
                              (n=8684, R²=0.8192), NEVER vs the shipped modeler_copy number
                              (n=7420, R²=0.8337) — switching DB alone drops R² ~0.014 purely
                              from the larger/different sample, so a modeler_copy comparison
                              would falsely red a healthy retrain. See output/r3_baseline/.
  G5  SQFT COVERAGE UP ...... the trained DB has MORE trainable (size_sqft>0) Rightmove rows
                              and FEWER active-no-sqft Rightmove rows than the pre-OCR baseline
                              (the whole point of the backfill); and n_samples rose.
  G6  TRAINED ON THE RIGHT DB (footgun guard) — the retrain must have read output/rentals.db
                              (the DB the backfill writes), NOT the stale modeler_copy default.
                              If the human forgets `--db output/rentals.db`, the 1,932-row
                              recovery is INVISIBLE and the model regresses to the old sample.

WHY SKIP-NOT-FAIL BEFORE THE RETRAIN:
  At write time the SHIPPED model (n_samples=7420, trained on modeler_copy) is still on disk.
  Running these gates against THAT artifact would falsely red G4/G5/G6 (it predates the
  backfill). So every gate that needs the *retrained* artifacts skips with a clear reason
  until the retrain is detected. Detection is EITHER:
    * env RETRAIN_DONE=1  (the human sets this when running the suite as the ship gate), OR
    * the meta shows it was trained on rentals.db with n_samples > the shipped 7420 baseline
      (i.e. a real retrain on the enriched superset already landed).
  Pure-logic gates (parser/threshold maths, baseline-file presence) run ALWAYS so the gate
  logic itself can't silently rot in PR CI.

CONCURRENCY: the retrain reads output/rentals.db and must run AFTER the backfill PID exits.
These gates only READ artifacts (pkl/json) + the DB in immutable mode — never write, never
trigger a train. They never touch the running backfill.

RUN (mid-backfill / pre-retrain — pure-logic gates run, artifact gates skip with reason):
    python3 -m pytest tests/test_retrain_gates.py -v
RUN (after `python3 retrain_canonical.py --db output/rentals.db`, as the SHIP gate):
    RETRAIN_DONE=1 python3 -m pytest tests/test_retrain_gates.py -v
"""
import json
import os
import pickle
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# ── Artifacts the retrain writes (atomically, in one run) ────────────────────
MODEL_PKL = ROOT / "output" / "rental_model_canonical.pkl"
FEATURES_PKL = ROOT / "output" / "rental_model_canonical_features.pkl"
META_JSON = ROOT / "output" / "rental_model_canonical_meta.json"
INFERENCE_JSON = ROOT / "output" / "rental_model_canonical_inference.json"
GOLDEN_JSON = ROOT / "output" / "feature_parity_golden.json"
API_MODEL = ROOT / "chrome-extension" / "api" / "model.json"
API_FEATURES = ROOT / "chrome-extension" / "api" / "features.json"
FIXTURE_DIFF = ROOT / "chrome-extension" / "fixture_diff.mjs"

# ── The DB the backfill writes AND the retrain MUST read (R1/R3 db-authority) ─
LIVE_DB = Path(os.environ.get("SCRAPER_DB_PATH", ROOT / "output" / "rentals.db"))
RETRAIN_DB_NAME = "rentals.db"  # the basename the meta.db_source must reference

# ── R3 BEFORE baselines (captured pre-OCR; the ONLY correct comparison) ──────
R3_DIR = ROOT / "output" / "r3_baseline"
BEFORE_RENTALSDB_TSV = R3_DIR / "BEFORE_rentalsdb_preOCR_8684.tsv"      # SAME-DB baseline (use this)
BEFORE_SHIPPED_TSV = R3_DIR / "BEFORE_shipped_modelercopy_7420.tsv"    # the SHIPPED number (do NOT gate on)
BEFORE_COVERAGE_TSV = R3_DIR / "BEFORE_sqft_coverage.tsv"

# Hard-coded fallbacks for the BEFORE rentals.db (pre-OCR) numbers, in case the TSV is
# moved. These are the captured ground-truth from output/r3_baseline/.
BEFORE_RENTALSDB_R2 = 0.8192
BEFORE_RENTALSDB_MAE = 1360.0
BEFORE_RENTALSDB_N = 8684
SHIPPED_N_SAMPLES = 7420  # the shipped model's modeler_copy sample size (the footgun anchor)

# Pre-OCR Rightmove sqft coverage baseline (from BEFORE_sqft_coverage.tsv).
BEFORE_RM_ACTIVE_SQFT = 3272
BEFORE_RM_ACTIVE_NOSQFT = 1932

# ── Ship gate thresholds (R3) ────────────────────────────────────────────────
R2_NONREGRESSION_BAND = 0.005   # NEW R² must be >= BEFORE(same DB) − 0.005
MAE_REGRESSION_BAND_GBP = 75.0  # NEW MAE must be <= BEFORE(same DB) + £75
EXPECTED_N_FEATURES = 150       # v20 schema: backfill adds DATA, never a feature

# Did the human flag the retrain as done? (matches test_backfill_acceptance.py convention)
RETRAIN_DONE_ENV = os.environ.get("RETRAIN_DONE", "").strip().lower() in {"1", "true", "yes"}


# ───────────────────────────── helpers ──────────────────────────────────────

def _immutable_uri(db_path: Path) -> str:
    return f"file:{db_path}?immutable=1"


def _db_available() -> bool:
    return LIVE_DB.exists() and LIVE_DB.stat().st_size > 0


def _read_meta() -> dict | None:
    if not META_JSON.exists():
        return None
    try:
        return json.loads(META_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _retrain_detected() -> tuple[bool, str]:
    """Did the *retrained-on-rentals.db* model land?

    True iff EITHER the human set RETRAIN_DONE=1, OR the on-disk meta already shows a
    retrain on the enriched superset (n_samples grew past the shipped 7420). The shipped
    artifact at write time is the modeler_copy model (n=7420) — that is NOT a retrain.
    """
    meta = _read_meta()
    if RETRAIN_DONE_ENV:
        return True, "RETRAIN_DONE=1 set by operator"
    if meta is None:
        return False, "no model meta on disk yet"
    n = meta.get("n_samples")
    if isinstance(n, int) and n > SHIPPED_N_SAMPLES:
        return True, f"meta n_samples={n} > shipped {SHIPPED_N_SAMPLES} (retrain on enriched superset detected)"
    return False, (
        f"shipped model still on disk (meta n_samples={n} == shipped {SHIPPED_N_SAMPLES}); "
        f"run `python3 retrain_canonical.py --db output/rentals.db` then re-run with RETRAIN_DONE=1"
    )


def _require_retrain():
    done, why = _retrain_detected()
    if not done:
        pytest.skip(f"retrain not done yet — {why}")


def _parse_overall_tsv(tsv: Path) -> dict:
    """Parse the OVERALL line of a _tail_error_eval MACHINE TSV → {n,R2,MAE,...}.

    Line shape: 'OVERALL<TAB>n=8684  features=150  R2=0.8192  MAE=£1,360  MAPE=17.8%  MedAPE=13.6%'
    """
    out: dict = {}
    for line in tsv.read_text().splitlines():
        if not line.startswith("OVERALL"):
            continue
        body = line.split("\t", 1)[1] if "\t" in line else line
        for tok in body.replace("£", "").replace(",", "").split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            v = v.rstrip("%")
            try:
                out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                out[k] = v
        break
    return out


def _parse_bands_tsv(tsv: Path) -> dict:
    """Parse BAND lines → {label: {n, bias_pct, bias_gbp, mape}}."""
    bands: dict = {}
    for line in tsv.read_text().splitlines():
        if not line.startswith("BAND"):
            continue
        parts = line.split("\t")
        label = parts[1]
        d: dict = {}
        for tok in parts[2:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        bands[label] = d
    return bands


def _compute_after_oof(db_path: Path):
    """Run the REAL tail-error OOF harness against the trained DB → (overall_str, band_rows).

    Reuses scripts/_tail_error_eval.py so the AFTER numbers are the exact training-path CV
    (KFold5 shuffle rs=42, log1p) the BEFORE baseline used — an apples-to-apples comparison.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib
    tee = importlib.import_module("scripts._tail_error_eval")  # may import as module if pkg
    return tee


# ───────────────────────── PURE-LOGIC GATES (always run) ─────────────────────
# These guard the gate machinery itself: the TSV parsers, the baseline files, and the
# comparison maths. If these rot, the real gates below would silently mis-judge a retrain.

@pytest.mark.unit
def test_before_baseline_files_exist():
    """R3 captured the BEFORE baselines; the gates depend on them. Fail loudly if missing."""
    assert R3_DIR.is_dir(), f"missing R3 baseline dir {R3_DIR}"
    assert BEFORE_RENTALSDB_TSV.exists(), (
        f"missing same-DB BEFORE baseline {BEFORE_RENTALSDB_TSV} — the ONLY correct R²/MAE "
        f"comparison for a rentals.db retrain"
    )
    assert BEFORE_COVERAGE_TSV.exists(), f"missing {BEFORE_COVERAGE_TSV}"


@pytest.mark.unit
def test_before_baseline_parses_to_known_numbers():
    """The TSV parser must reproduce R3's recorded BEFORE-rentals.db numbers exactly."""
    o = _parse_overall_tsv(BEFORE_RENTALSDB_TSV)
    assert o["n"] == BEFORE_RENTALSDB_N, o
    assert abs(o["R2"] - BEFORE_RENTALSDB_R2) < 1e-6, o
    assert abs(o["MAE"] - BEFORE_RENTALSDB_MAE) < 1.0, o
    assert o["features"] == EXPECTED_N_FEATURES, o


@pytest.mark.unit
def test_nonregression_band_maths():
    """Document + lock the comparison rule so a future edit can't loosen it silently.

    A retrain at exactly the BEFORE R² passes (>= BEFORE − band). One a hair below the
    band fails. This is the contract G4 enforces.
    """
    floor = BEFORE_RENTALSDB_R2 - R2_NONREGRESSION_BAND
    assert (BEFORE_RENTALSDB_R2) >= floor          # equal passes
    assert (BEFORE_RENTALSDB_R2 - 0.0049) >= floor  # within band passes
    assert not ((BEFORE_RENTALSDB_R2 - 0.0051) >= floor)  # beyond band fails


@pytest.mark.unit
def test_we_do_not_gate_on_the_shipped_modelercopy_number():
    """Guard against the R3 sample-confound trap.

    The shipped baseline (n=7420, R²=0.8337) is a DIFFERENT DB+sample than the retrain's
    (rentals.db). Gating AFTER-rentals.db against the 0.8337 shipped number would falsely
    red a healthy retrain (DB switch alone costs ~0.014 R²). Assert the gate's floor is
    derived from the SAME-DB baseline (0.8192), not the shipped one (0.8337).
    """
    shipped = _parse_overall_tsv(BEFORE_SHIPPED_TSV)["R2"] if BEFORE_SHIPPED_TSV.exists() else 0.8337
    same_db = _parse_overall_tsv(BEFORE_RENTALSDB_TSV)["R2"]
    assert abs(same_db - BEFORE_RENTALSDB_R2) < 1e-6
    # The floor we actually use must come from same_db, and must be BELOW the shipped number
    # (so we never accidentally demand the impossible 0.8337 of a rentals.db retrain).
    floor = same_db - R2_NONREGRESSION_BAND
    assert floor < shipped, (
        "gate floor is anchored to the shipped modeler_copy R² — that's the sample-confound "
        "trap; anchor to BEFORE_rentalsdb_preOCR instead"
    )


# ───────────────────────── G2 — FEATURE COUNT SANE ──────────────────────────

@pytest.mark.data
def test_gate_feature_count_is_150():
    """G2: the retrained model still has exactly the v20 150-feature schema.

    Checks the three places n_features lives stay consistent: features.pkl length,
    meta.n_features, and the model's own booster feature count.
    """
    _require_retrain()
    assert FEATURES_PKL.exists(), f"missing {FEATURES_PKL}"
    feats = pickle.loads(FEATURES_PKL.read_bytes())
    assert isinstance(feats, list) and len(feats) == EXPECTED_N_FEATURES, (
        f"feature list len={len(feats)} != {EXPECTED_N_FEATURES} — v20 schema changed; "
        f"the backfill must add DATA only, never a feature"
    )
    meta = _read_meta()
    assert meta and meta.get("n_features") == EXPECTED_N_FEATURES, meta

    model = pickle.loads(MODEL_PKL.read_bytes())
    try:
        booster_n = model.get_booster().num_features()
        assert booster_n == EXPECTED_N_FEATURES, f"booster num_features={booster_n}"
    except Exception:
        # Some xgboost versions expose n_features_in_ instead; accept either.
        n_in = getattr(model, "n_features_in_", None)
        assert n_in == EXPECTED_N_FEATURES, f"model n_features_in_={n_in}"


# ───────────────── G3 — NO STRING/OBJECT FEATURE (LEAKAGE GUARD) ─────────────

@pytest.mark.data
def test_gate_no_string_or_object_feature_selected():
    """G3: re-engineer the trained DB and assert NONE of the selected feature columns are
    object/string dtype.

    A string column in X is both an immediate train crash AND the classic accidental-leak
    vector (an id/address/target-derived label sneaking in). All 150 v20 features must be
    purely numeric (int/float). We re-engineer from a COPY of the live DB (immutable read →
    temp file) so the running backfill is never disturbed.
    """
    _require_retrain()
    if not _db_available():
        pytest.skip("no populated rentals.db")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import warnings
    warnings.filterwarnings("ignore")
    import retrain_canonical as rc
    import rental_price_models_v20 as v20

    # Copy via immutable backup so we never open the live DB for write and never block the
    # backfill. (sqlite3 .backup from an immutable-source connection.)
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "analysis_retrain_gate_g3.db"
    src = sqlite3.connect(_immutable_uri(LIVE_DB), uri=True)
    try:
        dst = sqlite3.connect(str(tmp))
        with dst:
            src.backup(dst)
        dst.close()
    finally:
        src.close()

    df = rc.load_recency_independent(str(tmp))
    df = v20.engineer_features_v20(df)
    feat_cols = v20.get_feature_columns_v20(df)
    assert len(feat_cols) == EXPECTED_N_FEATURES, len(feat_cols)

    offenders = [c for c in feat_cols if str(df[c].dtype) == "object"]
    assert not offenders, (
        f"object/string feature columns selected into X (leak/crash risk): {offenders}"
    )
    # And the trained model's feature list must equal the freshly-engineered selection.
    trained_feats = pickle.loads(FEATURES_PKL.read_bytes())
    assert list(trained_feats) == list(feat_cols), (
        "trained feature ORDER/SET diverges from a fresh engineer_features_v20 on the trained "
        "DB — schema drift between artifact and code"
    )


# ───────────────────────── G6 — TRAINED ON THE RIGHT DB ──────────────────────

@pytest.mark.data
def test_gate_trained_on_rentals_db_not_modeler_copy():
    """G6 (footgun guard): the retrain MUST have read output/rentals.db.

    If the human ran `retrain_canonical.py` with NO --db it silently used the stale
    modeler_copy default and the entire 1,932-row backfill is INVISIBLE — zero error, a
    quietly-worse model. We assert provenance two ways:
      (a) meta.db_source references 'rentals.db' (NOT 'modeler_copy' / 'merged'), AND
      (b) meta.n_samples > the shipped modeler_copy 7420 (a rentals.db retrain yields >=8684
          pre-OCR and MORE after recovery — it can never land AT 7420).
    """
    _require_retrain()
    meta = _read_meta()
    assert meta is not None, "no meta to verify provenance"
    src = str(meta.get("db_source", "")).lower()
    assert "modeler_copy" not in src and "merged" not in src, (
        f"meta.db_source={src!r} points at a STALE decoy DB — retrain must read rentals.db"
    )
    assert RETRAIN_DB_NAME in src, (
        f"meta.db_source={src!r} does not reference {RETRAIN_DB_NAME}; "
        f"run `python3 retrain_canonical.py --db output/rentals.db`"
    )
    n = meta.get("n_samples")
    assert isinstance(n, int) and n > SHIPPED_N_SAMPLES, (
        f"n_samples={n} did not exceed the shipped modeler_copy {SHIPPED_N_SAMPLES} — the "
        f"retrain likely read modeler_copy (the silent default), so the backfill is invisible"
    )


# ───────────────────────── G5 — SQFT COVERAGE UP ─────────────────────────────

@pytest.mark.data
def test_gate_sqft_coverage_increased():
    """G5: the trained DB recovered Rightmove sqft (the whole point of the backfill).

    Against the live DB (immutable read): active-no-sqft Rightmove rows must DROP below the
    pre-OCR baseline 1,932, trainable Rightmove sqft rows must RISE above 3,272, and the
    model's n_samples must exceed the shipped 7420. All recovered sqft must be sane (the
    retrain's own quality gate: 150<=sqft<=10000) so a garbage OCR value can't pad coverage.
    """
    _require_retrain()
    if not _db_available():
        pytest.skip("no populated rentals.db")

    conn = sqlite3.connect(_immutable_uri(LIVE_DB), uri=True)
    try:
        rm_active_sqft, = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE source='rightmove' AND is_active=1 "
            "AND size_sqft IS NOT NULL AND size_sqft > 0"
        ).fetchone()
        rm_active_nosqft, = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE source='rightmove' AND is_active=1 "
            "AND (size_sqft IS NULL OR size_sqft = 0)"
        ).fetchone()
        # Sanity: no recovered Rightmove sqft falls outside the retrain's training bounds.
        n_insane, = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE source='rightmove' AND size_sqft IS NOT NULL "
            "AND size_sqft > 0 AND (size_sqft < 150 OR size_sqft > 10000)"
        ).fetchone()
    finally:
        conn.close()

    assert rm_active_nosqft < BEFORE_RM_ACTIVE_NOSQFT, (
        f"active no-sqft Rightmove rows did not drop: {rm_active_nosqft} >= baseline "
        f"{BEFORE_RM_ACTIVE_NOSQFT} — backfill recovered nothing"
    )
    assert rm_active_sqft > BEFORE_RM_ACTIVE_SQFT, (
        f"trainable Rightmove sqft rows did not rise: {rm_active_sqft} <= baseline "
        f"{BEFORE_RM_ACTIVE_SQFT}"
    )
    assert n_insane == 0, (
        f"{n_insane} recovered Rightmove sqft values fall outside [150,10000] — bad OCR "
        f"would become bad training rows"
    )

    meta = _read_meta()
    assert meta and meta["n_samples"] > SHIPPED_N_SAMPLES, (
        f"n_samples {meta and meta.get('n_samples')} did not exceed shipped {SHIPPED_N_SAMPLES}"
    )


# ───────────────────── G4 — R²/MAE NOT REGRESSED (the helped gate) ───────────

@pytest.mark.data
def test_gate_r2_mae_not_regressed_vs_same_db_baseline():
    """G4: NEW CV R²/MAE must not regress vs the SAME-DB (rentals.db pre-OCR) baseline.

    Runs the REAL _tail_error_eval OOF harness on the live DB (immutable copy) — the exact
    KFold5/log1p training-path CV the BEFORE baseline used. Gate:
        NEW R²  >= BEFORE_rentalsdb_R2 − 0.005   (0.8192 − 0.005 = 0.8142)
        NEW MAE <= BEFORE_rentalsdb_MAE + £75    (1360 + 75 = £1,435)
    AND no price BAND's MAPE worsens materially (>1.0pp) — the recovered-sqft Rightmove rows
    cluster in the <2k/2-5k/5-10k bands; their tail error should improve or hold, never blow
    out. This is the backstop against a plausible-but-wrong OCR sqft poisoning a band.

    NOTE the asymmetry: we do NOT require R² to RISE. Recovering sqft for 1,932 rows makes
    them trainable but also adds harder-to-fit stock; the ship bar is NON-REGRESSION on the
    same DB, which is exactly what "the backfill didn't hurt + unlocked rows" means.
    """
    _require_retrain()
    if not _db_available():
        pytest.skip("no populated rentals.db")

    # Immutable copy so the OOF harness can open it lock-free without blocking the backfill.
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "analysis_retrain_gate_g4.db"
    src = sqlite3.connect(_immutable_uri(LIVE_DB), uri=True)
    try:
        dst = sqlite3.connect(str(tmp))
        with dst:
            src.backup(dst)
        dst.close()
    finally:
        src.close()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib
    import warnings
    warnings.filterwarnings("ignore")
    # _tail_error_eval lives in scripts/; import it as a top-level module.
    sys.path.insert(0, str(ROOT / "scripts"))
    tee = importlib.import_module("_tail_error_eval")

    import numpy as np
    from sklearn.metrics import mean_absolute_error, r2_score

    df, actual, preds, n_feats = tee.compute_oof(str(tmp))
    actual = np.asarray(actual, dtype=float)
    preds = np.asarray(preds, dtype=float)
    new_r2 = float(r2_score(actual, preds))
    new_mae = float(mean_absolute_error(actual, preds))

    assert n_feats == EXPECTED_N_FEATURES, f"OOF harness saw {n_feats} features"

    r2_floor = BEFORE_RENTALSDB_R2 - R2_NONREGRESSION_BAND
    assert new_r2 >= r2_floor, (
        f"R² REGRESSED: new={new_r2:.4f} < floor={r2_floor:.4f} "
        f"(BEFORE rentals.db={BEFORE_RENTALSDB_R2}); DO NOT SHIP"
    )
    mae_ceiling = BEFORE_RENTALSDB_MAE + MAE_REGRESSION_BAND_GBP
    assert new_mae <= mae_ceiling, (
        f"MAE REGRESSED: new=£{new_mae:,.0f} > ceiling=£{mae_ceiling:,.0f} "
        f"(BEFORE rentals.db=£{BEFORE_RENTALSDB_MAE:,.0f}); DO NOT SHIP"
    )

    # Per-band tail-error backstop: no band's MAPE may worsen by >1.0pp vs BEFORE rentals.db.
    before_bands = _parse_bands_tsv(BEFORE_RENTALSDB_TSV)
    after_band_rows = tee.band_table(actual, preds)  # [(label,n,bias_pct,bias_gbp,mape), ...]
    worsened = []
    for label, n, bias_pct, bias_gbp, mape in after_band_rows:
        if not n or label not in before_bands:
            continue
        before_mape = before_bands[label].get("mape")
        if before_mape is None:
            continue
        if mape - before_mape > 1.0:
            worsened.append((label, round(before_mape, 1), round(mape, 1)))
    assert not worsened, (
        f"price BAND tail-error WORSENED >1.0pp (before→after MAPE): {worsened}; DO NOT SHIP — "
        f"a plausible-but-wrong recovered sqft may be poisoning a band"
    )


# ───────────────────────── G1 — PARITY 0/0 (JS↔Python) ───────────────────────

@pytest.mark.data
@pytest.mark.parity
def test_gate_js_python_parity_zero_zero():
    """G1: node chrome-extension/fixture_diff.mjs must report 0 feature + 0 £ mismatches.

    This is the SHIP gate for the regen/ship sequence: after the human runs
        retrain_canonical → gen_inference_stats → gen_feature_parity_golden →
        canonical_predict.export_to_chrome → scripts/sync_js_freq_maps.py
    the JS predictor (chrome-extension/xgboost.js) must stay byte-equal to the Python
    engineer_features_v20 on the freshly-regenerated golden fixture, AND the £ estimate must
    match within 0.5%. Any drift here means the shipped extension diverges from the model.

    Requires node + the regenerated golden + exported api/ artifacts. Skips with a clear
    reason if the regen/ship steps haven't been run yet (the artifacts predate the retrain).
    """
    _require_retrain()
    import shutil
    if shutil.which("node") is None:
        pytest.skip("node not available — parity gate runs where node is installed")
    for p in (FIXTURE_DIFF, GOLDEN_JSON, API_MODEL, API_FEATURES):
        if not p.exists():
            pytest.skip(
                f"regen/ship artifact missing ({p.name}); run the post-retrain regen sequence "
                f"(gen_feature_parity_golden + canonical_predict.export_to_chrome + "
                f"sync_js_freq_maps) before the parity gate"
            )

    # Guard against a STALE golden: it must have been regenerated at/after the model pkl.
    if MODEL_PKL.exists() and GOLDEN_JSON.stat().st_mtime < MODEL_PKL.stat().st_mtime - 1:
        pytest.skip(
            "feature_parity_golden.json is OLDER than the retrained model pkl — regenerate it "
            "(gen_feature_parity_golden.py) before asserting parity"
        )

    proc = subprocess.run(
        ["node", str(FIXTURE_DIFF), str(GOLDEN_JSON)],
        cwd=str(ROOT / "chrome-extension"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = proc.stdout + "\n" + proc.stderr
    # fixture_diff.mjs exits 0 on PASS and prints the canonical summary line.
    assert proc.returncode == 0, f"parity FAILED (exit {proc.returncode}):\n{out}"
    assert "0 feature mismatch(es), 0 £ mismatch(es)" in out, (
        f"parity not 0/0 — JS↔Python drift after retrain:\n{out}"
    )
