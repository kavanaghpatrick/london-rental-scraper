/**
 * R6 + A8 — serving-query FIX gates (Wave 2 deferred prod-behavior sign-off).
 *
 * These were DOCUMENTING xfails (assert the desired/fixed behavior, mark XFAIL while the
 * bug was present). The prod-SQL fix has now SHIPPED, so the xfails are converted to hard
 * `check(...)` gates and the SQL-text proofs are rewritten to assert the FIXED shape.
 *
 *   R6 — no-space postcode dropped by SPLIT_PART (FIXED):
 *     The district gate was `SPLIT_PART(postcode, ' ', 1) = $district`. A listing stored
 *     with a NO-SPACE postcode 'SW34AJ' yields SPLIT_PART => 'SW34AJ' (no delimiter ->
 *     whole string) != 'SW3', so it was WRONGLY excluded. The fix normalizes the district
 *     via REPLACE(postcode,' ','') + an anchored outward-code regex (COALESCE, with
 *     SPLIT_PART retained as the legacy fallback). DESIRED & NOW ASSERTED: a district='SW3'
 *     query returns BOTH the spaced and the no-space peer. (rental + sale.)
 *
 *   A8 — cycle-relative last_seen freshness predicate (FIXED, FROZEN-SNAPSHOT-SAFE):
 *     The WHERE clause had ONLY `is_active = 1` — no last_seen predicate, so a 400-day-stale
 *     row that is still is_active=1 (a mark-inactive miss) was returned as a live comp. The
 *     fix adds a CYCLE-RELATIVE cutoff: last_seen >= (SELECT MAX(last_seen) FROM <table>) -
 *     INTERVAL '7 days' (NULL last_seen kept). CRITICAL: this is anchored to the DATA's own
 *     MAX, never wall-clock NOW() — a frozen snapshot (all last_seen ~= MAX) is NOT emptied.
 *     DESIRED & NOW ASSERTED: the 400-day-stale peer is excluded; a frozen snapshot returns
 *     ALL peers.
 *
 * NON-VACUOUS WITHOUT A DB: the SQL-TEXT proofs below ALWAYS run and assert the FIX is
 * STRUCTURALLY present (district gate normalizes compact postcodes; WHERE has a
 * cycle-relative MAX(last_seen) predicate and NOT a wall-clock NOW()-INTERVAL). If anyone
 * regresses the SQL shape, these text proofs trip first. The DB-backed gates run against the
 * Postgres SERVICE CONTAINER in CI (POSTGRES_TEST_URL); locally they SKIP the DB part (clean)
 * but keep the text proofs.
 *
 * Run: [POSTGRES_TEST_URL=postgres://…] node dashboard/test/serving_query_bug_doc_test.mjs
 */
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const LIB = join(__dirname, '..', 'src', 'lib');

const { buildSimilarQuery } = require(join(LIB, 'similarQuery.js'));
const { buildSaleSimilarQuery } = require(join(LIB, 'saleSimilarQuery.js'));

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK    ${name}`);
  else { failures++; console.log(`FAIL  ${name}${detail ? ' — ' + detail : ''}`); }
}

// --------------------------------------------------------------------------- //
// 1. SQL-TEXT proofs — ALWAYS run, no DB. Pin that the FIX is structurally present.
// --------------------------------------------------------------------------- //
function sqlTextProofs() {
  console.log('\n--- SQL-text proofs (always run; pin the FIX is structurally present) ---');
  const rental = buildSimilarQuery({ postcodeDistrict: 'SW3', bedrooms: 2, pricePcm: 5000, sizeSqft: 1000, propertyType: 'flat' }).text;
  const sale = buildSaleSimilarQuery({ postcodeDistrict: 'SW3', bedrooms: 2, askingPrice: 3_000_000, sizeSqft: 1000, propertyType: 'flat' }).text;

  // R6 FIXED: the district gate now normalizes compact (no-space) postcodes by stripping
  // the space and matching the anchored outward code, instead of the bare SPLIT_PART.
  const districtFixed = (sql) =>
    sql.includes("REPLACE(postcode, ' ', '')") && /\[A-Z\]\{1,2\}\[0-9\]\[0-9A-Z\]\?/.test(sql);
  check('R6 fixed: rental district gate normalizes compact postcodes', districtFixed(rental));
  check('R6 fixed: sale district gate normalizes compact postcodes', districtFixed(sale));
  // SPLIT_PART is retained as the COALESCE legacy fallback (keeps the structural-parity
  // token check green without editing similar_query_test.mjs).
  check('R6 fixed: rental keeps SPLIT_PART as the COALESCE fallback',
    rental.includes("SPLIT_PART(postcode, ' ', 1)"));
  check('R6 fixed: sale keeps SPLIT_PART as the COALESCE fallback',
    sale.includes("SPLIT_PART(postcode, ' ', 1)"));

  // A8 FIXED: the WHERE clause now has a CYCLE-RELATIVE last_seen freshness predicate
  // anchored to MAX(last_seen), and NOT a wall-clock NOW()-INTERVAL cutoff.
  const hasLastSeenPredicate = (sql) => /last_seen\s*(>=|>|BETWEEN)|last_seen[^,\n]*INTERVAL/i.test(sql);
  check('A8 fixed: rental query still gates on is_active = 1', rental.includes('is_active = 1'));
  check('A8 fixed: rental has a cycle-relative MAX(last_seen) freshness predicate',
    hasLastSeenPredicate(rental) && /MAX\(\s*last_seen/i.test(rental));
  check('A8 fixed: sale has a cycle-relative MAX(last_seen) freshness predicate',
    hasLastSeenPredicate(sale) && /MAX\(\s*last_seen/i.test(sale));
  // String-pin (twin of test_workflow_sql_is_cycle_relative_not_wallclock): NEVER wall-clock.
  check('A8 frozen-snapshot guard: rental SQL does NOT use wall-clock NOW() - INTERVAL',
    !/NOW\(\)\s*-\s*INTERVAL/i.test(rental));
  check('A8 frozen-snapshot guard: sale SQL does NOT use wall-clock NOW() - INTERVAL',
    !/NOW\(\)\s*-\s*INTERVAL/i.test(sale));
}

// --------------------------------------------------------------------------- //
// 2. DB-backed FIX confirmation — runs against the Postgres service container in CI.
// --------------------------------------------------------------------------- //
const RENTAL_SCHEMA = `
CREATE TABLE IF NOT EXISTS listings (
  id SERIAL PRIMARY KEY, source TEXT NOT NULL, property_id TEXT NOT NULL, url TEXT,
  address TEXT, postcode TEXT, price_pcm INTEGER, size_sqft INTEGER, bedrooms INTEGER,
  property_type TEXT, is_active INTEGER DEFAULT 1, last_seen TEXT, UNIQUE(source, property_id)
);`;
const SALE_SCHEMA = `
CREATE TABLE IF NOT EXISTS sale_listings (
  id SERIAL PRIMARY KEY, source TEXT NOT NULL, property_id TEXT NOT NULL, url TEXT,
  address TEXT, postcode TEXT, asking_price BIGINT, size_sqft INTEGER, bedrooms INTEGER,
  property_type TEXT, is_active INTEGER DEFAULT 1, is_under_offer INTEGER DEFAULT 0,
  last_seen TEXT, UNIQUE(source, property_id)
);`;

// A fixed "today" for the A8 staleness seed. The A8 cutoff is CYCLE-RELATIVE (anchored to
// MAX(last_seen) in the data), so the absolute date here is irrelevant — only the SPREAD
// between rows matters. The fresh peers sit at daysAgo(1) (=> MAX), and the STALE peer at
// daysAgo(400) is ~399 days behind MAX, well past the 7-day window.
function daysAgo(n) {
  const d = new Date('2026-06-22T09:00:00Z');
  d.setUTCDate(d.getUTCDate() - n);
  return d.toISOString().slice(0, 19).replace('T', ' ');
}

async function rentalDbFix(client) {
  console.log('\n--- R6/A8 rental DB confirmation ---');
  await client.query('DROP TABLE IF EXISTS listings');
  await client.query(RENTAL_SCHEMA);

  const subject = { postcodeDistrict: 'SW3', bedrooms: 2, pricePcm: 5000, sizeSqft: 1000, propertyType: 'flat' };
  const ins = async (pid, postcode, lastSeen) =>
    client.query(
      `INSERT INTO listings (source, property_id, postcode, price_pcm, size_sqft, bedrooms,
         property_type, is_active, last_seen, address, url)
       VALUES ('savills',$1,$2,5100,1000,2,'flat',1,$3,'A St','http://x/'||$1)`,
      [pid, postcode, lastSeen]
    );

  // R6 seed: a no-space 'SW34AJ' and a spaced 'SW3 4AJ' peer; both fresh.
  await ins('SPACED', 'SW3 4AJ', daysAgo(1));
  await ins('NOSPACE', 'SW34AJ', daysAgo(1));
  // A8 seed: a 400-day-stale (still is_active=1) spaced SW3 peer.
  await ins('STALE', 'SW3 9ZZ', daysAgo(400));

  const { text, values } = buildSimilarQuery(subject);
  const { rows } = await client.query(text, values);
  const ids = new Set(rows.map((r) => r.property_id));

  // The spaced peer is correctly returned (control — proves the seed is otherwise valid).
  check('R6 control: spaced SW3 4AJ peer IS returned', ids.has('SPACED'), [...ids].join(','));

  // R6 FIXED (hard gate): the no-space 'SW34AJ' peer is now returned.
  check('R6 rental: no-space SW34AJ peer matches district SW3', ids.has('NOSPACE'), [...ids].join(','));

  // A8 FIXED (hard gate): the 400-day-stale peer is now excluded (cycle-relative cutoff).
  check('A8 rental: 400-day-stale is_active=1 peer is excluded', !ids.has('STALE'), [...ids].join(','));
}

async function saleDbFix(client) {
  console.log('\n--- R6/A8 sale DB confirmation ---');
  await client.query('DROP TABLE IF EXISTS sale_listings');
  await client.query(SALE_SCHEMA);

  const subject = { postcodeDistrict: 'SW3', bedrooms: 2, askingPrice: 3_000_000, sizeSqft: 1000, propertyType: 'flat' };
  const ins = async (pid, postcode, lastSeen) =>
    client.query(
      `INSERT INTO sale_listings (source, property_id, postcode, asking_price, size_sqft,
         bedrooms, property_type, is_active, is_under_offer, last_seen, address, url)
       VALUES ('savills',$1,$2,3100000,1000,2,'flat',1,0,$3,'A St','http://x/'||$1)`,
      [pid, postcode, lastSeen]
    );

  await ins('SPACED', 'SW3 4AJ', daysAgo(1));
  await ins('NOSPACE', 'SW34AJ', daysAgo(1));
  await ins('STALE', 'SW3 9ZZ', daysAgo(400));

  const { text, values } = buildSaleSimilarQuery(subject);
  const { rows } = await client.query(text, values);
  const ids = new Set(rows.map((r) => r.property_id));

  check('R6 control: spaced SW3 4AJ sale peer IS returned', ids.has('SPACED'), [...ids].join(','));
  check('R6 sale: no-space SW34AJ sale peer matches district SW3', ids.has('NOSPACE'), [...ids].join(','));
  check('A8 sale: 400-day-stale is_active=1 sale peer is excluded', !ids.has('STALE'), [...ids].join(','));
}

// --------------------------------------------------------------------------- //
// 3. FROZEN-SNAPSHOT SAFETY (the highest-risk gate): a snapshot whose rows ALL share one
//    old last_seen must NOT be emptied. A cycle-relative (MAX-7d) cutoff returns everything;
//    a wall-clock NOW()-7d filter would return 0 and FAIL — this is the empty-peers guard.
// --------------------------------------------------------------------------- //
async function frozenSnapshotNotEmptied(client) {
  console.log('\n--- A8 frozen-snapshot safety (rental) ---');
  await client.query('DROP TABLE IF EXISTS listings');
  await client.query(RENTAL_SCHEMA);
  const subject = { postcodeDistrict: 'SW3', bedrooms: 2, pricePcm: 5000, sizeSqft: 1000, propertyType: 'flat' };
  const ins = async (pid, lastSeen) => client.query(
    `INSERT INTO listings (source, property_id, postcode, price_pcm, size_sqft, bedrooms,
       property_type, is_active, last_seen, address, url)
     VALUES ('savills',$1,'SW3 4AJ',5100,1000,2,'flat',1,$2,'A St','http://x/'||$1)`, [pid, lastSeen]);
  // ALL peers share one OLD last_seen (90 days ago) — a frozen snapshot, all is_active=1.
  for (const pid of ['F1', 'F2', 'F3', 'F4', 'F5']) await ins(pid, daysAgo(90));
  const { text, values } = buildSimilarQuery(subject);
  const { rows } = await client.query(text, values);
  // A cycle-relative cutoff (MAX-7d) leaves a frozen snapshot fully intact. A wall-clock
  // NOW()-7d filter would return 0 here and FAIL — this is the empty-peers regression guard.
  check('A8 frozen-snapshot (rental): all 5 old-but-uniform peers returned (NOT emptied)', rows.length === 5, `got ${rows.length}`);

  // NULL-last_seen-kept: a peer with last_seen=NULL must be returned (treated as fresh) even
  // alongside a fresh cohort that advances MAX.
  console.log('\n--- A8 NULL-last_seen kept (rental) ---');
  await client.query('DROP TABLE IF EXISTS listings');
  await client.query(RENTAL_SCHEMA);
  for (const pid of ['G1', 'G2']) await ins(pid, daysAgo(1)); // fresh cohort -> MAX is recent
  await client.query(
    `INSERT INTO listings (source, property_id, postcode, price_pcm, size_sqft, bedrooms,
       property_type, is_active, last_seen, address, url)
     VALUES ('savills','GNULL','SW3 4AJ',5100,1000,2,'flat',1,NULL,'A St','http://x/GNULL')`);
  const r2 = await client.query(text, values);
  const ids2 = new Set(r2.rows.map((r) => r.property_id));
  check('A8 NULL-last_seen peer IS returned (treated as fresh)', ids2.has('GNULL'), [...ids2].join(','));
}

async function frozenSnapshotNotEmptiedSale(client) {
  console.log('\n--- A8 frozen-snapshot safety (sale) ---');
  await client.query('DROP TABLE IF EXISTS sale_listings');
  await client.query(SALE_SCHEMA);
  const subject = { postcodeDistrict: 'SW3', bedrooms: 2, askingPrice: 3_000_000, sizeSqft: 1000, propertyType: 'flat' };
  const ins = async (pid, lastSeen) => client.query(
    `INSERT INTO sale_listings (source, property_id, postcode, asking_price, size_sqft, bedrooms,
       property_type, is_active, is_under_offer, last_seen, address, url)
     VALUES ('savills',$1,'SW3 4AJ',3100000,1000,2,'flat',1,0,$2,'A St','http://x/'||$1)`, [pid, lastSeen]);
  // ALL peers share one OLD last_seen (90 days ago) — a frozen sale snapshot, all is_active=1.
  for (const pid of ['F1', 'F2', 'F3', 'F4', 'F5']) await ins(pid, daysAgo(90));
  const { text, values } = buildSaleSimilarQuery(subject);
  const { rows } = await client.query(text, values);
  check('A8 frozen-snapshot (sale): all 5 old-but-uniform sale peers returned (NOT emptied)', rows.length === 5, `got ${rows.length}`);

  // NULL-last_seen-kept (sale).
  console.log('\n--- A8 NULL-last_seen kept (sale) ---');
  await client.query('DROP TABLE IF EXISTS sale_listings');
  await client.query(SALE_SCHEMA);
  for (const pid of ['G1', 'G2']) await ins(pid, daysAgo(1));
  await client.query(
    `INSERT INTO sale_listings (source, property_id, postcode, asking_price, size_sqft, bedrooms,
       property_type, is_active, is_under_offer, last_seen, address, url)
     VALUES ('savills','GNULL','SW3 4AJ',3100000,1000,2,'flat',1,0,NULL,'A St','http://x/GNULL')`);
  const r2 = await client.query(text, values);
  const ids2 = new Set(r2.rows.map((r) => r.property_id));
  check('A8 NULL-last_seen sale peer IS returned (treated as fresh)', ids2.has('GNULL'), [...ids2].join(','));
}

async function main() {
  sqlTextProofs();

  const url = process.env.POSTGRES_TEST_URL || process.env.DATABASE_URL;
  if (!url) {
    console.log('\nSKIP DB-backed FIX confirmation: set POSTGRES_TEST_URL (or DATABASE_URL) ' +
      'to a Postgres instance to run it. The SQL-text proofs above already pin the fix ' +
      'structurally. (CI sets this via a Postgres service container.)');
    process.exit(failures ? 1 : 0);
  }

  const { Client } = require('pg');
  const client = new Client({ connectionString: url });
  await client.connect();
  try {
    await rentalDbFix(client);
    await saleDbFix(client);
    await frozenSnapshotNotEmptied(client);
    await frozenSnapshotNotEmptiedSale(client);
  } finally {
    await client.end();
  }

  if (failures) {
    console.log(`\n=== FAIL: ${failures} serving-query FIX gate(s) failed. ===`);
    process.exit(1);
  }
  console.log(`\n=== PASS: R6 (no-space-postcode) + A8 (cycle-relative last_seen, frozen-snapshot-safe) ` +
    `FIX gates green (rental + sale). ===`);
}

main().catch((e) => {
  console.error('serving_query_bug_doc_test crashed:', e);
  process.exit(1);
});
