/**
 * ANTI-SILENT-SKIP GUARD (node side).
 *
 * The team's core distrust: a test that passes locally but silently STOPS RUNNING in
 * CI — renamed, removed, un-wired, or typo'd out of the workflow — looks identical to
 * "all green". The extension harnesses were a live example: similar_properties_test.mjs
 * and content_load_test.mjs passed green for weeks while wired into NO workflow, giving
 * zero PR-CI protection for the comps location hard-gate and the content.js load fix.
 *
 * This guard makes that failure class LOUD. It asserts two invariants and exits non-zero
 * (failing the CI job) if either breaks:
 *
 *   1. COMPLETENESS — every harness in REQUIRED_HARNESSES exists on disk AND is actually
 *      invoked by .github/workflows/ci.yml. If someone deletes a `node …` line, or
 *      renames a harness without updating the workflow, this fails — the test can't
 *      silently vanish from CI.
 *
 *   2. NO ORPHANS — every runnable harness file on disk (`*_test.mjs` + fixture_diff.mjs,
 *      excluding the extract_test_shim.mjs helper and this guard) is referenced by at
 *      least one workflow. A NEW harness that someone forgets to wire in fails here, so
 *      "I wrote a test" can't quietly mean "the test never runs in CI".
 *
 * Run: node chrome-extension/ci_harness_guard.mjs   (exit 0 = pass, 1 = fail)
 */
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, basename } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const WORKFLOWS = join(ROOT, '.github', 'workflows');

// The harnesses CI MUST run. This list is the source of truth for "what protects the
// extension on every PR". Adding a harness means adding it here AND to ci.yml.
const REQUIRED_HARNESSES = [
  'fixture_diff.mjs',            // JS↔Python £/feature parity (also enforced standalone)
  'rightmove_extract_test.mjs',
  'knightfrank_extract_test.mjs',
  'foxtons_extract_test.mjs',
  'floorplan_discovery_test.mjs',
  'foxtons_ocr_skip_test.mjs',
  'similar_properties_test.mjs', // comps location HARD-gate (findSimilarProperties)
  'content_load_test.mjs',       // content.js load + log-recursion + SPA-retry guards
  // Inc4 — For Sale Fair Value mode on real "to buy" detail pages (per-site detection +
  // extraction, Chestertons /sales/ unblock, structural POA fork, shared sale OCR, rental
  // regression). These are RED until the content.js writer lands the Inc4 fix.
  'rightmove_sale_extract_test.mjs',
  'foxtons_sale_extract_test.mjs',
  'chestertons_sale_extract_test.mjs',
  'savills_sale_extract_test.mjs',
  'knightfrank_sale_extract_test.mjs',
  'structural_sale_poa_test.mjs',
  'rental_regression_guard_test.mjs',
  'sale_fixture_diff.mjs',       // SALE JS↔Python £/feature parity (Inc4a). NOTE: ends in
                                 // neither '_test.mjs' nor '== fixture_diff.mjs', so the
                                 // orphan glob (Invariant 2) does NOT match it — REQUIRED_HARNESSES
                                 // membership is the ONLY thing pinning it to disk + ci.yml, and is
                                 // sufficient. Do NOT widen the orphan glob (that would edit the
                                 // rental-guard behavior for a sale-only file).
];

// Helpers / non-standalone files that are imported, not invoked as a `node …` harness.
const NOT_STANDALONE = new Set([
  'extract_test_shim.mjs',       // shared browser shim, imported by the extract tests
  'ci_harness_guard.mjs',        // this file
]);

let failures = 0;
const fail = (msg) => { failures++; console.log(`FAIL ${msg}`); };
const ok = (msg) => console.log(`OK   ${msg}`);

// --- Load workflow text (ci.yml is the gate; scan all workflows for orphan check) ----
const ciYml = readFileSync(join(WORKFLOWS, 'ci.yml'), 'utf8');
let allWorkflowText = '';
for (const f of readdirSync(WORKFLOWS).filter((n) => n.endsWith('.yml') || n.endsWith('.yaml'))) {
  allWorkflowText += '\n' + readFileSync(join(WORKFLOWS, f), 'utf8');
}

// --- Invariant 1: every REQUIRED harness exists on disk AND is invoked in ci.yml ------
const onDisk = new Set(readdirSync(__dirname).filter((n) => n.endsWith('.mjs')));
for (const h of REQUIRED_HARNESSES) {
  if (!onDisk.has(h)) {
    fail(`required harness missing on disk: chrome-extension/${h}`);
    continue;
  }
  // Match `node chrome-extension/<h>` (or bare `<h>`) anywhere in ci.yml.
  const referenced = ciYml.includes(h);
  if (referenced) ok(`ci.yml invokes ${h}`);
  else fail(`required harness NOT invoked by ci.yml (silently dropped from CI?): ${h}`);
}

// --- Invariant 2: no runnable harness on disk is orphaned (wired into no workflow) ----
for (const f of onDisk) {
  if (NOT_STANDALONE.has(f)) continue;
  const isRunnable = f.endsWith('_test.mjs') || f === 'fixture_diff.mjs';
  if (!isRunnable) continue;
  if (allWorkflowText.includes(f)) {
    ok(`harness wired into a workflow: ${f}`);
  } else {
    fail(`ORPHAN harness — exists but invoked by NO workflow (zero CI protection): ${f}. ` +
         `Wire it into ci.yml's node-parity job (and add to REQUIRED_HARNESSES) or remove it.`);
  }
}

if (failures) {
  console.log(`\n=== FAIL: ${failures} harness-wiring problem(s). A critical extension test ` +
              `may have silently stopped running in CI. ===`);
  process.exit(1);
}
console.log(`\n=== PASS: all ${REQUIRED_HARNESSES.length} required harnesses are wired into ci.yml; ` +
            `no orphaned harnesses. ===`);
