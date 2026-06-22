/**
 * ANTI-SILENT-SKIP GUARD for the dashboard route harnesses (Workstream A).
 *
 * Mirrors chrome-extension/ci_harness_guard.mjs for dashboard/test/*. Two invariants;
 * exits non-zero (failing the CI job) if either breaks:
 *
 *   1. WIRING — every required route harness exists on disk AND is invoked by
 *      .github/workflows/ci.yml's dashboard-routes job. A renamed/removed/un-wired
 *      harness fails here instead of silently vanishing from CI.
 *
 *   2. NO SILENT DB-SKIP — the /api/similar harness SKIPS (exit 0) when no
 *      POSTGRES_TEST_URL is set, so a dev without Postgres isn't blocked. But in CI the
 *      Postgres SERVICE CONTAINER guarantees POSTGRES_TEST_URL is set. This guard
 *      asserts POSTGRES_TEST_URL is present AND reachable, so the similar harness can't
 *      quietly skip its DB assertions while the job still goes green. (The exact
 *      "green-but-not-actually-testing" failure class the team distrusts.)
 *
 * Run (in CI, after the harnesses): node dashboard/test/dashboard_routes_guard.mjs
 */
import { readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const CI_YML = join(ROOT, '.github', 'workflows', 'ci.yml');

// The dashboard route harnesses CI MUST run. Source of truth for "what protects the
// serving routes on every PR". Adding one means adding it here AND to ci.yml.
const REQUIRED_HARNESSES = [
  'predict_estimate_test.mjs',      // /api/predict real v20 estimate (no DB)
  'similar_query_test.mjs',         // /api/similar real query vs service-container Postgres
  'predict_sale_estimate_test.mjs', // /api/predict-sale real sale_v1 estimate (Inc4a; no DB)
  'similar_sale_query_test.mjs',    // /api/similar-sale real query vs Postgres (Inc4a; SSTC+empty)
  // --- Wave 2 (Group SERVING): the route-HANDLER + dual-SQL parity + query-bug harnesses.
  // Pinned here so they can't be dropped from ci.yml's dashboard-routes job and silently
  // stop protecting the serving routes. Membership here is what makes Invariant 1b's
  // orphan check pass for these *_test.mjs files (they MUST also appear in ci.yml).
  'route_handler_test.mjs',         // A6 — invoke the real RENTAL Next.js handlers (400/CORS/503/500-vs-empty)
  'route_handler_sale_test.mjs',    // A6 — invoke the real FOR-SALE handlers (sale-specific 400/503 divergences)
  'dual_sql_equality_test.mjs',     // A7 — db.ts↔similarQuery.js + saleDb.ts↔saleSimilarQuery.js byte-equality
  'serving_query_bug_doc_test.mjs', // R6/A8 — no-space-postcode + stale-peer xfail documenting tests
];

// Helper/guard files that are NOT standalone `node …` harnesses.
// _route_loader.mjs is the Wave-2 A6 helper (transpiles + loads route.ts, imported by the
// route-handler harnesses) — it exports a function and is not invoked as a `node …`
// harness, so it is excluded from the orphan check (it also lacks the `_test.mjs` suffix).
const NOT_STANDALONE = new Set(['dashboard_routes_guard.mjs', '_route_loader.mjs']);

let failures = 0;
const fail = (m) => { failures++; console.log(`FAIL ${m}`); };
const ok = (m) => console.log(`OK   ${m}`);

const ciYml = readFileSync(CI_YML, 'utf8');
const onDisk = new Set(readdirSync(__dirname).filter((n) => n.endsWith('.mjs')));

// --- Invariant 1: every required harness exists + is invoked in ci.yml ----------------
for (const h of REQUIRED_HARNESSES) {
  if (!onDisk.has(h)) { fail(`required route harness missing on disk: dashboard/test/${h}`); continue; }
  if (ciYml.includes(h)) ok(`ci.yml invokes ${h}`);
  else fail(`required route harness NOT invoked by ci.yml (silently dropped?): ${h}`);
}

// --- Invariant 1b: no orphan harness on disk (wired into no workflow) -----------------
for (const f of onDisk) {
  if (NOT_STANDALONE.has(f)) continue;
  if (!f.endsWith('_test.mjs')) continue;
  if (ciYml.includes(f)) ok(`harness wired into ci.yml: ${f}`);
  else fail(`ORPHAN route harness — on disk but invoked by NO workflow: ${f}`);
}

// --- Invariant 2: the DB-backed harness must NOT have silently skipped in CI ----------
const url = process.env.POSTGRES_TEST_URL || process.env.DATABASE_URL;
if (!url) {
  fail(
    'POSTGRES_TEST_URL is unset — the /api/similar harness would SKIP its DB ' +
    'assertions, leaving the route untested while the job stays green. CI must provide ' +
    'a Postgres service container. (Set POSTGRES_TEST_URL to run this guard locally.)'
  );
} else {
  ok('POSTGRES_TEST_URL is set (similar harness ran its DB assertions)');
  // Reachability: open a real connection so a misconfigured/unhealthy container fails
  // here rather than masquerading as a passing similar harness.
  try {
    const { Client } = require('pg');
    const c = new Client({ connectionString: url });
    await c.connect();
    const r = await c.query('SELECT 1 AS ok');
    await c.end();
    if (r.rows[0].ok === 1) ok('Postgres service container is reachable (SELECT 1)');
    else fail('Postgres reachable but SELECT 1 returned unexpected result');
  } catch (e) {
    fail(`Postgres at POSTGRES_TEST_URL not reachable: ${e.message}`);
  }
}

if (failures) {
  console.log(`\n=== FAIL: ${failures} dashboard-route harness-wiring/DB problem(s). ===`);
  process.exit(1);
}
console.log(`\n=== PASS: route harnesses wired into ci.yml + Postgres reachable. ===`);
