/**
 * REAL behavioral test for /api/similar's query logic against a REAL Postgres.
 *
 * WHY THIS SHAPE (resolving the @vercel/postgres-vs-CI-Postgres blocker):
 *   The route's DB client (@vercel/postgres -> @neondatabase/serverless) cannot talk
 *   to a plain local Postgres — its query path always goes through the neon HTTP
 *   `fetchEndpoint`, so a service-container Postgres would need a fragile neon proxy.
 *   Instead we test the route's ACTUAL query (the single-source similarQuery.js that
 *   db.ts imports) against a real Postgres via the plain `pg` driver. Same SQL, same
 *   params, same stats helper the route runs -> real behavioral coverage with no proxy,
 *   no next runtime, no TS transpile. In CI this runs against a Postgres SERVICE
 *   CONTAINER (see .github/workflows/ci.yml dashboard-db job).
 *
 * It also runs a STRUCTURAL PARITY check: the positional SQL in similarQuery.js must
 * carry the same scoring constants + WHERE clauses as db.ts's tagged-template query,
 * so the tested module can't silently drift from the production route.
 *
 * Connection: reads POSTGRES_TEST_URL (or DATABASE_URL). If neither is set, it SKIPS
 * with exit 0 and a clear message (so a dev without Postgres isn't blocked) — CI sets
 * it, so CI runs it for real. The ci_harness_guard / job wiring make sure it isn't
 * silently dropped.
 *
 * Run: POSTGRES_TEST_URL=postgres://... node dashboard/test/similar_query_test.mjs
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const LIB = join(__dirname, '..', 'src', 'lib');

const { buildSimilarQuery, computeSimilarStats } = require(join(LIB, 'similarQuery.js'));

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) {
    console.log(`OK   ${name}`);
  } else {
    failures++;
    console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`);
  }
}

// --------------------------------------------------------------------------- //
// 0. STRUCTURAL PARITY: similarQuery.js must match db.ts's template scoring.
//    (Runs with NO Postgres — pure text check — so it always protects against drift.)
// --------------------------------------------------------------------------- //
function structuralParity() {
  const dbts = readFileSync(join(LIB, 'db.ts'), 'utf8');
  const { text } = buildSimilarQuery({
    postcodeDistrict: 'SW3',
    bedrooms: 2,
    pricePcm: 5000,
    sizeSqft: 1000,
    propertyType: 'flat',
  });
  // Scoring constants + the load-bearing clauses must appear in BOTH.
  const tokens = [
    'THEN 0.30', 'THEN 0.15', 'THEN 0.25', 'THEN 0.10', 'THEN 0.07', 'THEN 0.05',
    "source IN ('savills', 'knightfrank')",
    "source IN ('chestertons', 'foxtons')",
    'is_active = 1',
    "SPLIT_PART(postcode, ' ', 1)",
    'similarity_score > 0.3',
    'LIMIT 15',
  ];
  for (const t of tokens) {
    check(
      `structural parity: db.ts and similarQuery.js both contain ${JSON.stringify(t)}`,
      dbts.includes(t) && text.includes(t),
      `db.ts=${dbts.includes(t)} module=${text.includes(t)}`
    );
  }
}

// --------------------------------------------------------------------------- //
// Schema (subset the query touches) + deterministic seed.
// --------------------------------------------------------------------------- //
const SCHEMA = `
CREATE TABLE IF NOT EXISTS listings (
  id SERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  property_id TEXT NOT NULL,
  url TEXT,
  address TEXT,
  postcode TEXT,
  price_pcm INTEGER,
  size_sqft INTEGER,
  bedrooms INTEGER,
  property_type TEXT,
  is_active INTEGER DEFAULT 1,
  last_seen TEXT,
  UNIQUE(source, property_id)
);
`;

// Subject: 2-bed, ~1000 sqft, ~£5000pcm flat in SW3. Seed a mix:
//   - 3 strong SW3 peers (should score > 0.3 and be returned)
//   - 1 wrong-district (SW1) peer at the same price (district gate -> excluded)
//   - 1 inactive SW3 peer (is_active=0 -> excluded)
//   - 1 far-price SW3 row outside the 0.5x-2x band (excluded by price range)
//   - 1 row to be excluded by excludeId
const SEED = [
  // source, property_id, postcode, price_pcm, size_sqft, bedrooms, property_type, is_active
  ['savills',     'S1', 'SW3 5RA', 5200, 1050, 2, 'flat',  1],  // strong peer (source bonus)
  ['knightfrank', 'K1', 'SW3 4CD', 5800, 1120, 2, 'flat',  1],  // strong peer
  ['foxtons',     'F1', 'SW3 3DZ', 4800,  980, 2, 'flat',  1],  // strong peer
  ['rightmove',   'R1', 'SW1W 9DA', 5100, 1000, 2, 'flat', 1],  // WRONG district -> excluded
  ['savills',     'S2', 'SW3 6EA', 5000, 1010, 2, 'flat',  0],  // INACTIVE -> excluded
  ['savills',     'S3', 'SW3 7AA', 15000, 1000, 2, 'flat', 1],  // price 3x -> out of range
  ['foxtons',     'F2', 'SW3 2AB', 5050, 1005, 2, 'flat',  1],  // strong peer, used for excludeId
];

async function seed(client) {
  await client.query('DROP TABLE IF EXISTS listings');
  await client.query(SCHEMA);
  for (const [source, pid, postcode, price, sqft, beds, ptype, active] of SEED) {
    await client.query(
      `INSERT INTO listings (source, property_id, postcode, price_pcm, size_sqft,
         bedrooms, property_type, is_active, last_seen, address, url)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`,
      [source, pid, postcode, price, sqft, beds, ptype, active,
       '2026-05-20 09:00:00', `${pid} Some Street`, `https://example/${pid}`]
    );
  }
}

async function runQuery(client, params) {
  const { text, values } = buildSimilarQuery(params);
  const { rows } = await client.query(text, values);
  return rows;
}

async function behavioral(client) {
  const subject = {
    postcodeDistrict: 'SW3',
    bedrooms: 2,
    pricePcm: 5000,
    sizeSqft: 1000,
    propertyType: 'flat',
  };

  // --- core: seeded SW3 peers come back, wrong/inactive/out-of-range excluded ---
  const rows = await runQuery(client, subject);
  const ids = rows.map((r) => r.property_id).sort();
  check('returns the 4 strong SW3 peers', ids.join(',') === 'F1,F2,K1,S1', `got ${ids.join(',')}`);
  check('excludes wrong-district SW1 peer (R1)', !ids.includes('R1'));
  check('excludes inactive SW3 peer (S2)', !ids.includes('S2'));
  check('excludes out-of-price-range SW3 row (S3)', !ids.includes('S3'));
  check('all returned rows are in the SW3 district',
    rows.every((r) => String(r.postcode).split(' ')[0] === 'SW3'),
    rows.map((r) => r.postcode).join('|'));
  check('every returned row scores > 0.3',
    rows.every((r) => Number(r.similarity_score) > 0.3),
    rows.map((r) => r.similarity_score).join(','));
  check('ppsf is computed for sized rows',
    rows.every((r) => r.ppsf === null || Number(r.ppsf) > 0));

  // --- stats: computeSimilarStats (the route's helper) on the real rows ---
  const stats = computeSimilarStats(rows, subject);
  check('stats.peer_count matches row count', stats.peer_count === rows.length, `${stats.peer_count} vs ${rows.length}`);
  check('stats.avg_price is a positive integer', Number.isInteger(stats.avg_price) && stats.avg_price > 0, `${stats.avg_price}`);
  check('stats.min_price <= avg_price <= max_price',
    stats.min_price <= stats.avg_price && stats.avg_price <= stats.max_price,
    `${stats.min_price}/${stats.avg_price}/${stats.max_price}`);
  check('stats.your_percentile in [0,100]', stats.your_percentile >= 0 && stats.your_percentile <= 100, `${stats.your_percentile}`);

  // --- excludeId removes the named property from the result ---
  const excluded = await runQuery(client, { ...subject, excludeId: 'F2' });
  check('excludeId removes F2 from peers', !excluded.map((r) => r.property_id).includes('F2'),
    excluded.map((r) => r.property_id).join(','));
  check('excludeId returns one fewer peer', excluded.length === rows.length - 1, `${excluded.length} vs ${rows.length}`);

  // --- empty result: a district with no listings returns zero peers + zeroed stats ---
  const none = await runQuery(client, { ...subject, postcodeDistrict: 'E1' });
  check('unknown district returns no peers', none.length === 0, `${none.length}`);
  const noneStats = computeSimilarStats(none, { ...subject, postcodeDistrict: 'E1' });
  check('empty result -> peer_count 0', noneStats.peer_count === 0);
  check('empty result -> your_percentile defaults to 50', noneStats.your_percentile === 50, `${noneStats.your_percentile}`);
  check('empty result -> avg_ppsf null', noneStats.avg_ppsf === null);

  // --- bedroom range: a 4-bed subject (range 3-5) returns NONE of the 2-bed seed ---
  const fourBed = await runQuery(client, { ...subject, bedrooms: 4 });
  check('4-bed subject excludes all 2-bed peers (range gate)', fourBed.length === 0, `${fourBed.length}`);
}

async function main() {
  // Structural parity always runs (no DB).
  structuralParity();

  const url = process.env.POSTGRES_TEST_URL || process.env.DATABASE_URL;
  if (!url) {
    console.log('\nSKIP behavioral DB tests: set POSTGRES_TEST_URL (or DATABASE_URL) to a ' +
      'Postgres instance to run them. Structural parity ran above. (CI sets this via a ' +
      'Postgres service container.)');
    // Exit non-zero ONLY if structural parity failed; otherwise a clean skip.
    process.exit(failures ? 1 : 0);
  }

  const { Client } = require('pg');
  const client = new Client({ connectionString: url });
  await client.connect();
  try {
    await seed(client);
    await behavioral(client);
  } finally {
    await client.end();
  }

  if (failures) {
    console.log(`\n=== FAIL: ${failures} /api/similar query check(s) failed. ===`);
    process.exit(1);
  }
  console.log('\n=== PASS: /api/similar query behaves correctly against real Postgres. ===');
}

main().catch((e) => {
  console.error('similar_query_test crashed:', e);
  process.exit(1);
});
