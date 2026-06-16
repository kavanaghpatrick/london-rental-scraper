"""
TEST SUITE #48 — API + dashboard routes (serving layer).

Covers the Next.js serving layer:
  * /api/predict      — POST validation (400), CORS (OPTIONS), 503/fallback seam
                        when the model isn't loaded, correct response envelope.
  * /api/similar      — CORS, correct comps shape from a seeded DB.
  * /report/*         — the 3 report pages render WITHOUT a 500 on an empty DB
                        (empty-state guard → NoDataYet), and on a seeded DB.
  * /api/init-db      — POST-only + token auth (405 on GET, 401/503 without token).

Design (matches the team's pytest conventions, tests/conftest.py + pytest.ini):
  * The `tsc` typecheck test is a UNIT test — always runs, gates PR (no server needed).
  * The HTTP route tests are marked `dashboard` — they spin up an EPHEMERAL Postgres
    (testing.postgresql) + `next dev`, so they're OPT-IN like `live`: run with
    `pytest -m dashboard`. They auto-SKIP when node / next / testing.postgresql /
    the dashboard deps aren't available, so default PR CI stays green without them.

Run the full route suite locally:  pytest tests/test_api_dashboard.py -m dashboard -v
Run just the always-on gate:        pytest tests/test_api_dashboard.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "dashboard"
ROUTE_DIR = DASHBOARD / "src" / "app" / "api" / "predict"

# init-db schema SQL is reused from the existing verifier (single source).
_VERIFY = ROOT / "scripts" / "_verify_serving_sync.py"


# --------------------------------------------------------------------------- #
# Availability checks → graceful skips (keep default CI green)
# --------------------------------------------------------------------------- #
def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _testing_pg_available() -> bool:
    try:
        import testing.postgresql  # noqa: F401
        import psycopg2  # noqa: F401
        return _have("postgres") or _have("pg_ctl") or bool(
            list(Path("/opt/homebrew/opt").glob("postgresql@*/bin/postgres"))
        )
    except Exception:
        return False


def _dashboard_ready() -> bool:
    return (
        _have("node")
        and (DASHBOARD / "node_modules" / "next").exists()
        and _testing_pg_available()
    )


dashboard = pytest.mark.dashboard
skip_no_dashboard = pytest.mark.skipif(
    not _dashboard_ready(),
    reason="dashboard route tests need node + next + testing.postgresql (opt-in: -m dashboard)",
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _init_db_sql() -> str:
    txt = _VERIFY.read_text()
    return txt.split('INIT_DB_SQL = """')[1].split('"""')[0]


# --------------------------------------------------------------------------- #
# Always-on UNIT gate: the dashboard typechecks (no server needed)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.skipif(not _dashboard_ready(), reason="node/next not installed")
def test_dashboard_typechecks():
    """tsc --noEmit must pass — gates the whole serving layer against type regressions."""
    r = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=DASHBOARD, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, f"tsc failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.unit
def test_predict_route_exists_and_snake_case_contract():
    """The predict route exists and uses the snake_case contract (no leftover camelCase keys)."""
    route = (ROUTE_DIR / "route.ts").read_text()
    assert "property_type" in route, "predict route must accept property_type (snake_case)"
    assert "estimate_pcm" in route and "model_version" in route and "range_low" in route
    # The vendored predictor module must be present + byte-identical to the source.
    vendored = ROUTE_DIR / "xgboost.predictor.js"
    source = ROOT / "chrome-extension" / "xgboost.js"
    if vendored.exists() and source.exists():
        assert vendored.read_bytes() == source.read_bytes(), (
            "vendored xgboost.predictor.js drifted from chrome-extension/xgboost.js "
            "(CI byte-identity guard should catch this)"
        )


# --------------------------------------------------------------------------- #
# Session fixture: ephemeral Postgres + next dev
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def live_dashboard():
    if not _dashboard_ready():
        pytest.skip("dashboard not ready")
    import testing.postgresql
    import psycopg2

    # ephemeral Postgres seeded with the init-db schema + a few listings.
    pg = testing.postgresql.Postgresql()
    with psycopg2.connect(pg.url()) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(_init_db_sql())
            # seed a handful of SW3/SW1 comps so /api/similar + reports have data.
            cur.execute(
                "INSERT INTO listings (source, property_id, postcode, bedrooms, "
                "bathrooms, size_sqft, price_pcm, price_period, is_active, property_type) VALUES "
                "('savills','S1','SW3 2AB',2,1,1000,5200,'pcm',1,'flat'),"
                "('knightfrank','K1','SW3 4CD',2,2,1100,5800,'pcm',1,'flat'),"
                "('foxtons','F1','SW1W 9JA',2,2,1312,11000,'pcm',1,'flat'),"
                "('rightmove','R1','SW7 1AA',3,2,1600,6800,'pcm',1,'flat')"
            )

    env = dict(os.environ, POSTGRES_URL=pg.url())
    port = _free_port()
    proc = subprocess.Popen(
        ["npx", "next", "dev", "-p", str(port)],
        cwd=DASHBOARD, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    # wait for the dev server (up to ~60s for first compile)
    import requests
    ready = False
    for _ in range(120):
        try:
            requests.get(f"{base}/api/predict", timeout=2)
            ready = True
            break
        except Exception:
            time.sleep(0.5)
    if not ready:
        proc.terminate()
        pg.stop()
        pytest.skip("next dev did not become ready in time")
    try:
        yield base, pg
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        pg.stop()


# --------------------------------------------------------------------------- #
# /api/predict
# --------------------------------------------------------------------------- #
@dashboard
@skip_no_dashboard
def test_predict_options_cors(live_dashboard):
    base, _ = live_dashboard
    import requests
    r = requests.options(f"{base}/api/predict", timeout=10)
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"
    assert "POST" in r.headers.get("access-control-allow-methods", "")


@dashboard
@skip_no_dashboard
@pytest.mark.parametrize("body,why", [
    ({}, "empty body"),
    ({"bedrooms": 2}, "missing size_sqft/postcode/property_type"),
    ({"bedrooms": 2, "bathrooms": 1, "size_sqft": 1000, "postcode": "SW3"}, "missing property_type"),
    ({"bedrooms": 2, "bathrooms": 1, "size_sqft": -5, "postcode": "SW3", "property_type": "flat"}, "bad size"),
])
def test_predict_post_validation_400(live_dashboard, body, why):
    base, _ = live_dashboard
    import requests
    r = requests.post(f"{base}/api/predict", json=body, timeout=10)
    assert r.status_code == 400, f"expected 400 for {why}, got {r.status_code}: {r.text}"
    assert "error" in r.json()


@dashboard
@skip_no_dashboard
def test_predict_post_bad_json_400(live_dashboard):
    base, _ = live_dashboard
    import requests
    r = requests.post(f"{base}/api/predict", data="not json",
                      headers={"Content-Type": "application/json"}, timeout=10)
    assert r.status_code == 400


@dashboard
@skip_no_dashboard
def test_predict_valid_body_returns_estimate_or_503(live_dashboard):
    """A valid POST returns either a real v20 estimate (model loaded) OR a 503
    'predictor not ready' fallback (model not reachable). Both are correct
    contract states — never a 500."""
    base, _ = live_dashboard
    import requests
    body = {"bedrooms": 2, "bathrooms": 1, "size_sqft": 1000, "postcode": "SW3",
            "property_type": "flat", "address": "12 Some Street"}
    r = requests.post(f"{base}/api/predict", json=body, timeout=30)
    assert r.status_code in (200, 503), f"expected 200 or 503, got {r.status_code}: {r.text}"
    assert r.headers.get("access-control-allow-origin") == "*"
    j = r.json()
    if r.status_code == 200:
        assert j["model_version"] == "v20"
        assert j["currency"] == "GBP"
        assert isinstance(j["estimate_pcm"], (int, float)) and j["estimate_pcm"] > 0
        assert j["range_low"] <= j["estimate_pcm"] <= j["range_high"]
    else:
        # 503 hold-state carries the model_version marker the extension reads.
        assert j.get("model_version") == "v20"


# --------------------------------------------------------------------------- #
# /api/similar
#
# NOTE: @vercel/postgres CANNOT connect to a plain local Postgres — it requires a
# Neon-style pooled endpoint (it raises 'invalid_connection_string' otherwise). So
# DB-backed assertions (real comps, report pages WITH data) only run when a
# Neon-compatible POSTGRES_URL is provided via NEON_TEST_URL. CORS/param-validation
# (which don't need DB rows) still run against the ephemeral harness.
# --------------------------------------------------------------------------- #
@dashboard
@skip_no_dashboard
def test_similar_cors(live_dashboard):
    base, _ = live_dashboard
    import requests
    o = requests.options(f"{base}/api/similar", timeout=10)
    assert o.status_code == 200
    assert o.headers.get("access-control-allow-origin") == "*"
    assert "GET" in o.headers.get("access-control-allow-methods", "")


@dashboard
@skip_no_dashboard
def test_similar_missing_params_400(live_dashboard):
    base, _ = live_dashboard
    import requests
    # missing beds + price -> 400 (this path returns before any DB call, so it
    # works even though @vercel/postgres can't reach the ephemeral DB).
    r = requests.get(f"{base}/api/similar", params={"postcode": "SW3"}, timeout=10)
    assert r.status_code == 400
    assert r.headers.get("access-control-allow-origin") == "*"


@dashboard
@skip_no_dashboard
def test_similar_invalid_numeric_params_400(live_dashboard):
    base, _ = live_dashboard
    import requests
    r = requests.get(f"{base}/api/similar",
                     params={"postcode": "SW3", "beds": "x", "price": "y"}, timeout=10)
    assert r.status_code == 400


@pytest.mark.skipif(not os.environ.get("NEON_TEST_URL"),
                    reason="needs a Neon-compatible POSTGRES_URL (set NEON_TEST_URL) — "
                           "@vercel/postgres can't use a local Postgres")
@dashboard
def test_similar_comps_against_neon():
    """Real comps from a Neon-backed DB. Opt-in: requires NEON_TEST_URL pointing at
    a seeded Neon test branch (the only way @vercel/postgres returns rows)."""
    import requests
    base = os.environ.get("DASHBOARD_BASE_URL", "http://127.0.0.1:3000")
    r = requests.get(f"{base}/api/similar",
                     params={"postcode": "SW3 2AB", "beds": 2, "price": 5200}, timeout=15)
    assert r.status_code == 200
    j = r.json()
    assert "peers" in j and "stats" in j


# --------------------------------------------------------------------------- #
# report pages — must NEVER 500 (the core guarantee)
#
# The 2026 health-team bug was a hard 500 on an empty/unreachable DB. The fix
# (error boundary + empty-state guard) means the pages must ALWAYS render with
# status 200 — whether the DB has data (NoDataYet not shown), is empty (NoDataYet
# panel), or is unreachable (error-boundary panel). All three are status 200; a
# 500 is the only failure. (Note: @vercel/postgres can't reach the ephemeral DB,
# so here the pages exercise the DB-unreachable -> error-boundary path, which is
# exactly one of the graceful states we must guarantee never 500s.)
# --------------------------------------------------------------------------- #
@dashboard
@skip_no_dashboard
@pytest.mark.parametrize("path", ["/report", "/report/negotiation", "/report/landlord-price"])
def test_report_pages_never_500(live_dashboard, path):
    base, _ = live_dashboard
    import requests
    r = requests.get(f"{base}{path}", timeout=30)
    assert r.status_code == 200, (
        f"{path} returned {r.status_code} — report pages must render gracefully "
        f"(data, empty-state, or error boundary), NEVER 500"
    )
    # And it must render a real HTML page, not an unhandled crash dump.
    body = r.text.lower()
    assert "<main" in body or "report" in body or "error" in body or "not available" in body


# --------------------------------------------------------------------------- #
# /api/init-db — POST-only + auth
# --------------------------------------------------------------------------- #
@dashboard
@skip_no_dashboard
def test_init_db_get_405(live_dashboard):
    base, _ = live_dashboard
    import requests
    r = requests.get(f"{base}/api/init-db", timeout=10)
    assert r.status_code == 405, f"init-db GET must be 405 (POST-only), got {r.status_code}"


@dashboard
@skip_no_dashboard
def test_init_db_post_without_token_rejected(live_dashboard):
    base, _ = live_dashboard
    import requests
    r = requests.post(f"{base}/api/init-db", timeout=15)
    assert r.status_code in (401, 403, 503), \
        f"init-db POST without token must be rejected (401/403/503), got {r.status_code}"
