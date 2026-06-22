/**
 * R6 + A8 — serving-query bug DOCUMENTING tests (Wave 2, Group SERVING).
 *
 * These pin two KNOWN, currently-unfixed /api/similar(-sale) query bugs. The prod-SQL fix
 * is DEFERRED to a separate prod-behavior sign-off, so here we DOCUMENT the bug with an
 * xfail: we assert the DESIRED (fixed) behavior and mark it an EXPECTED FAILURE while the
 * bug is present. The harness stays GREEN as long as the bug is still there, and FLIPS RED
 * (XPASS) the moment the bug is fixed — that XPASS is the signal to delete the xfail and
 * convert it into a hard gate.
 *
 *   R6 — no-space postcode dropped by SPLIT_PART:
 *     The district gate is `SPLIT_PART(postcode, ' ', 1) = $district`. A listing stored with
 *     a NO-SPACE postcode 'SW34AJ' yields SPLIT_PART => 'SW34AJ' (no delimiter -> whole
 *     string) != 'SW3', so it is WRONGLY excluded. A correctly-spaced 'SW3 4AJ' peer in the
 *     same building IS returned. DESIRED: a district='SW3' query returns BOTH. CURRENT: only
 *     the spaced one. (rental + sale.)
 *
 *   A8 — no last_seen freshness predicate:
 *     db.ts comments "Only include listings seen in the last 7 days to avoid showing
 *     stale/removed listings", but the WHERE clause has ONLY `is_active = 1` — no last_seen
 *     predicate. A 400-day-stale row that is still is_active=1 (e.g. a mark-inactive miss)
 *     is returned as a live comp. DESIRED: the 400-day-stale peer is excluded. CURRENT: it
 *     appears.
 *
 * NON-VACUOUS WITHOUT A DB: the SQL-TEXT proofs below ALWAYS run and assert the bug is
 * STRUCTURALLY present (district gate uses bare SPLIT_PART; WHERE has is_active but no
 * last_seen predicate). If a prod fix changes the SQL shape, these text proofs trip first.
 * The DB-backed xfail confirmation runs against the Postgres SERVICE CONTAINER in CI
 * (POSTGRES_TEST_URL); locally it SKIPS the DB part (clean) but keeps the text proofs.
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
let xfailCount = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK    ${name}`);
  else { failures++; console.log(`FAIL  ${name}${detail ? ' — ' + detail : ''}`); }
}
/**
 * Document a deferred bug. `desiredCond` is the FIXED behavior:
 *   - desiredCond FALSE  => XFAIL (bug still present, as expected) — stays green.
 *   - desiredCond TRUE   => XPASS (bug appears fixed) — FAILS, so the maintainer converts
 *                            this xfail into a real assertion (the flip-to-hard-gate signal).
 */
function xfail(name, desiredCond, reason) {
  if (!desiredCond) {
    xfailCount++;
    console.log(`XFAIL ${name} — deferred: ${reason}`);
  } else {
    failures++;
    console.log(`XPASS ${name} — the documented bug appears FIXED. Convert this xfail into a ` +
      `hard assertion (remove xfail) and wire the prod-SQL fix sign-off. (${reason})`);
  }
}

// --------------------------------------------------------------------------- //
// 1. SQL-TEXT proofs — ALWAYS run, no DB. Pin that the bug is structurally present.
// --------------------------------------------------------------------------- //
function sqlTextProofs() {
  console.log('\n--- SQL-text proofs (always run; pin the bug is structurally present) ---');
  const rental = buildSimilarQuery({ postcodeDistrict: 'SW3', bedrooms: 2, pricePcm: 5000, sizeSqft: 1000, propertyType: 'flat' }).text;
  const sale = buildSaleSimilarQuery({ postcodeDistrict: 'SW3', bedrooms: 2, askingPrice: 3_000_000, sizeSqft: 1000, propertyType: 'flat' }).text;

  // R6: the district gate uses a bare SPLIT_PART on a single-space delimiter — the exact
  // construct that drops no-space postcodes.
  check('R6 proof: rental district gate uses SPLIT_PART single-space delimiter',
    rental.includes("SPLIT_PART(postcode, ' ', 1) = $11"));
  check('R6 proof: sale district gate uses SPLIT_PART single-space delimiter',
    sale.includes("SPLIT_PART(postcode, ' ', 1) = $11"));

  // A8: the WHERE clause gates on is_active but has NO last_seen freshness predicate.
  const hasLastSeenPredicate = (sql) => /last_seen\s*(>=|>|BETWEEN)|last_seen[^,\n]*INTERVAL/i.test(sql);
  check('A8 proof: rental query gates on is_active = 1', rental.includes('is_active = 1'));
  check('A8 proof: rental query has NO last_seen freshness predicate (the bug)',
    !hasLastSeenPredicate(rental));
  check('A8 proof: sale query has NO last_seen freshness predicate (the bug)',
    !hasLastSeenPredicate(sale));
}

// --------------------------------------------------------------------------- //
// 2. DB-backed xfail confirmation — runs against the Postgres service container in CI.
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

// A fixed "today" for the A8 staleness seed (the SQL has no date dependence today, but a
// FIXED fix would; we seed an explicitly 400-day-old last_seen so the desired predicate,
// once added, would exclude it).
function daysAgo(n) {
  const d = new Date('2026-06-22T09:00:00Z');
  d.setUTCDate(d.getUTCDate() - n);
  return d.toISOString().slice(0, 19).replace('T', ' ');
}

async function rentalDbXfail(client) {
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

  // R6 DESIRED (deferred): the no-space 'SW34AJ' peer SHOULD also be returned.
  xfail('R6 rental: no-space SW34AJ peer matches district SW3',
    ids.has('NOSPACE'),
    "no-space postcode dropped by SPLIT_PART(postcode, ' ', 1); fix deferred to prod-SQL sign-off");

  // A8 DESIRED (deferred): the 400-day-stale peer SHOULD be excluded.
  xfail('A8 rental: 400-day-stale is_active=1 peer is excluded',
    !ids.has('STALE'),
    'no last_seen freshness predicate (only is_active=1); fix deferred to prod-SQL sign-off');
}

async function saleDbXfail(client) {
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
  xfail('R6 sale: no-space SW34AJ sale peer matches district SW3',
    ids.has('NOSPACE'),
    "no-space postcode dropped by SPLIT_PART(postcode, ' ', 1); fix deferred to prod-SQL sign-off");
  xfail('A8 sale: 400-day-stale is_active=1 sale peer is excluded',
    !ids.has('STALE'),
    'no last_seen freshness predicate (only is_active=1); fix deferred to prod-SQL sign-off');
}

async function main() {
  sqlTextProofs();

  const url = process.env.POSTGRES_TEST_URL || process.env.DATABASE_URL;
  if (!url) {
    console.log('\nSKIP DB-backed xfail confirmation: set POSTGRES_TEST_URL (or DATABASE_URL) ' +
      'to a Postgres instance to run it. The SQL-text proofs above already pin the bug ' +
      'structurally. (CI sets this via a Postgres service container.)');
    process.exit(failures ? 1 : 0);
  }

  const { Client } = require('pg');
  const client = new Client({ connectionString: url });
  await client.connect();
  try {
    await rentalDbXfail(client);
    await saleDbXfail(client);
  } finally {
    await client.end();
  }

  if (failures) {
    console.log(`\n=== FAIL: ${failures} documenting check(s) failed (an XPASS means a deferred ` +
      `bug is fixed — convert its xfail to a hard gate). ===`);
    process.exit(1);
  }
  console.log(`\n=== PASS: ${xfailCount} deferred serving-query bug(s) documented as XFAIL ` +
    `(R6 no-space-postcode + A8 stale-peer; flip to hard gates when the prod-SQL fix ships). ===`);
}

main().catch((e) => {
  console.error('serving_query_bug_doc_test crashed:', e);
  process.exit(1);
});
