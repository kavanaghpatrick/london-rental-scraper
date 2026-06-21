/**
 * REAL behavioral test for /api/similar-sale's query logic against a REAL Postgres.
 *
 * WHY THIS SHAPE (sale analogue of similar_query_test.mjs, same @vercel/postgres-vs-CI
 * blocker resolution):
 *   The route's DB client (@vercel/postgres -> @neondatabase/serverless) cannot talk to a
 *   plain local Postgres — its query path always goes through the neon HTTP fetchEndpoint.
 *   Instead we test the route's ACTUAL query (the single-source saleSimilarQuery.js that
 *   saleDb.ts imports) against a real Postgres via the plain `pg` driver. Same SQL, same
 *   params, same stats helper the route runs -> real behavioral coverage with no proxy, no
 *   next runtime, no TS transpile. In CI this runs against a Postgres SERVICE CONTAINER.
 *
 * It also runs a STRUCTURAL PARITY check: the positional SQL in saleSimilarQuery.js must
 * carry the same SALE-SPECIFIC clauses as saleDb.ts's tagged-template query (sale_listings
 * table, asking_price column, is_active = 1, the SSTC/under-offer exclusion, district gate,
 * similarity threshold, LIMIT 15), so the tested module can't silently drift from the
 * production route.
 *
 * GRACEFUL-EMPTY (the Inc4a/Inc4b seam): the route+query+test ship green in Inc4a, but the
 * PRODUCTION sale_listings rows in prod Neon are Inc4b. So this harness ALSO proves that an
 * EMPTY (truncated) sale_listings table yields 0 peers + the empty-default stats
 * (your_percentile=50, avg_ppsf=null) — that degrade-to-empty is what lets Inc4a ship
 * before Inc4b's data exists.
 *
 * Connection: reads POSTGRES_TEST_URL (or DATABASE_URL). If neither is set, the DB-backed
 * asserts SKIP with exit 0 and a clear message (so a dev without Postgres isn't blocked) —
 * CI sets it via a Postgres service container, so CI runs them for real. The structural
 * parity asserts ALWAYS run (no DB needed). The dashboard_routes_guard forces the URL set
 * in CI so the DB asserts can't silently vanish.
 *
 * NOTE: this harness depends on Group B artifacts (dashboard/src/lib/saleSimilarQuery.js +
 * saleDb.ts). It is EXPECTED to be RED until Group B lands those files; that is the TDD red
 * phase.
 *
 * Run: POSTGRES_TEST_URL=postgres://... node dashboard/test/similar_sale_query_test.mjs
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const LIB = join(__dirname, '..', 'src', 'lib');

// Group B file — RED until Group B lands saleSimilarQuery.js.
const { buildSaleSimilarQuery, computeSaleSimilarStats } =
  require(join(LIB, 'saleSimilarQuery.js'));

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
// 0. STRUCTURAL PARITY: saleSimilarQuery.js must match saleDb.ts's template.
//    ALWAYS runs (no Postgres — pure text check) so it always protects against drift,
//    including in a no-DB dev environment.
// --------------------------------------------------------------------------- //
function structuralParity() {
  const saleDbTs = readFileSync(join(LIB, 'saleDb.ts'), 'utf8'); // Group B file — RED until lands.
  const { text } = buildSaleSimilarQuery({
    postcodeDistrict: 'SW3',
    bedrooms: 2,
    askingPrice: 3_000_000,
    sizeSqft: 1000,
    propertyType: 'flat',
  });
  // SALE-SPECIFIC load-bearing clauses must appear in BOTH the lib query text and the
  // saleDb.ts tagged template.
  const tokens = [
    'sale_listings',                 // sale table (NOT listings)
    'asking_price',                  // sale price column (NOT price_pcm)
    'is_active = 1',                 // active gate
    'is_under_offer',                // SSTC/under-offer exclusion (sale-specific)
    "SPLIT_PART(postcode, ' ', 1)",  // district gate
    'similarity_score > 0.3',        // similarity threshold
    'LIMIT 15',                      // result cap
  ];
  for (const t of tokens) {
    check(
      `structural parity: saleDb.ts and saleSimilarQuery.js both contain ${JSON.stringify(t)}`,
      saleDbTs.includes(t) && text.includes(t),
      `saleDb.ts=${saleDbTs.includes(t)} module=${text.includes(t)}`
    );
  }
}

// --------------------------------------------------------------------------- //
// Schema (subset the query touches) + deterministic seed. Sale-scale prices (£M).
// --------------------------------------------------------------------------- //
const SCHEMA = `
CREATE TABLE IF NOT EXISTS sale_listings (
  id SERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  property_id TEXT NOT NULL,
  url TEXT,
  address TEXT,
  postcode TEXT,
  asking_price BIGINT,
  size_sqft INTEGER,
  bedrooms INTEGER,
  property_type TEXT,
  is_active INTEGER DEFAULT 1,
  is_under_offer INTEGER DEFAULT 0,
  last_seen TEXT,
  UNIQUE(source, property_id)
);
`;

// Subject: 2-bed, ~1000 sqft, ~£3M flat in SW3. Seed a mix:
//   - 3 strong SW3 peers (should score > 0.3 and be returned)
//   - 1 wrong-district (SW1) peer at the same price (district gate -> excluded)
//   - 1 inactive SW3 peer (is_active=0 -> excluded)
//   - 1 far-price SW3 row outside the 0.5x-2x band (excluded by price range)
//   - 1 is_under_offer=1 SW3 peer (SSTC -> excluded by the sale-specific clause)
//   - 1 row to be excluded by excludeId
const SEED = [
  // source, property_id, postcode, asking_price, size_sqft, beds, ptype, is_active, is_under_offer
  ['savills',     'S1', 'SW3 5RA', 3_200_000, 1050, 2, 'flat', 1, 0],  // strong peer (source bonus)
  ['knightfrank', 'K1', 'SW3 4CD', 3_600_000, 1120, 2, 'flat', 1, 0],  // strong peer
  ['foxtons',     'F1', 'SW3 3DZ', 2_800_000,  980, 2, 'flat', 1, 0],  // strong peer
  ['rightmove',   'R1', 'SW1W 9DA', 3_100_000, 1000, 2, 'flat', 1, 0], // WRONG district -> excluded
  ['savills',     'S2', 'SW3 6EA', 3_000_000, 1010, 2, 'flat', 0, 0],  // INACTIVE -> excluded
  ['savills',     'S3', 'SW3 7AA', 9_000_000, 1000, 2, 'flat', 1, 0],  // price 3x -> out of range
  ['knightfrank', 'U1', 'SW3 8BB', 3_050_000, 1015, 2, 'flat', 1, 1],  // UNDER OFFER (SSTC) -> excluded
  ['foxtons',     'F2', 'SW3 2AB', 3_050_000, 1005, 2, 'flat', 1, 0],  // strong peer, used for excludeId
];

async function seed(client) {
  await client.query('DROP TABLE IF EXISTS sale_listings');
  await client.query(SCHEMA);
  for (const [source, pid, postcode, price, sqft, beds, ptype, active, underOffer] of SEED) {
    await client.query(
      `INSERT INTO sale_listings (source, property_id, postcode, asking_price, size_sqft,
         bedrooms, property_type, is_active, is_under_offer, last_seen, address, url)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`,
      [source, pid, postcode, price, sqft, beds, ptype, active, underOffer,
       '2026-05-20 09:00:00', `${pid} Some Street`, `https://example/${pid}`]
    );
  }
}

async function runQuery(client, params) {
  const { text, values } = buildSaleSimilarQuery(params);
  const { rows } = await client.query(text, values);
  return rows;
}

async function behavioral(client) {
  const subject = {
    postcodeDistrict: 'SW3',
    bedrooms: 2,
    askingPrice: 3_000_000,
    sizeSqft: 1000,
    propertyType: 'flat',
  };

  // --- core: seeded SW3 peers come back; wrong/inactive/out-of-range/SSTC excluded ---
  const rows = await runQuery(client, subject);
  const ids = rows.map((r) => r.property_id).sort();
  check('returns the 4 strong SW3 sale peers', ids.join(',') === 'F1,F2,K1,S1', `got ${ids.join(',')}`);
  check('excludes wrong-district SW1 peer (R1)', !ids.includes('R1'));
  check('excludes inactive SW3 peer (S2)', !ids.includes('S2'));
  check('excludes out-of-price-range SW3 row (S3)', !ids.includes('S3'));
  // The SALE-SPECIFIC assertion: an under-offer (SSTC) comp must NOT appear.
  check('excludes SSTC / under-offer SW3 peer (U1)', !ids.includes('U1'), `got ${ids.join(',')}`);
  check('all returned rows are in the SW3 district',
    rows.every((r) => String(r.postcode).split(' ')[0] === 'SW3'),
    rows.map((r) => r.postcode).join('|'));
  check('every returned row scores > 0.3',
    rows.every((r) => Number(r.similarity_score) > 0.3),
    rows.map((r) => r.similarity_score).join(','));
  check('ppsf is computed for sized rows',
    rows.every((r) => r.ppsf === null || Number(r.ppsf) > 0));

  // --- stats: computeSaleSimilarStats (the route's helper) on the real rows ---
  const stats = computeSaleSimilarStats(rows, subject);
  check('stats.peer_count matches row count', stats.peer_count === rows.length, `${stats.peer_count} vs ${rows.length}`);
  check('stats.avg_price is a positive number', Number(stats.avg_price) > 0, `${stats.avg_price}`);
  check('stats.min_price <= avg_price <= max_price',
    stats.min_price <= stats.avg_price && stats.avg_price <= stats.max_price,
    `${stats.min_price}/${stats.avg_price}/${stats.max_price}`);
  check('stats.your_percentile in [0,100]', stats.your_percentile >= 0 && stats.your_percentile <= 100, `${stats.your_percentile}`);

  // --- excludeId removes the named property from the result ---
  const excluded = await runQuery(client, { ...subject, excludeId: 'F2' });
  check('excludeId removes F2 from peers', !excluded.map((r) => r.property_id).includes('F2'),
    excluded.map((r) => r.property_id).join(','));
  check('excludeId returns one fewer peer', excluded.length === rows.length - 1, `${excluded.length} vs ${rows.length}`);

  // --- bedroom range: a 4-bed subject (range 3-5) returns NONE of the 2-bed seed ---
  const fourBed = await runQuery(client, { ...subject, bedrooms: 4 });
  check('4-bed subject excludes all 2-bed peers (range gate)', fourBed.length === 0, `${fourBed.length}`);

  // --- GRACEFUL-EMPTY: a truncated sale_listings table -> 0 peers + empty-default stats ---
  // (proves Inc4a degrades to empty before Inc4b's prod data exists.)
  await client.query('TRUNCATE sale_listings');
  const empty = await runQuery(client, subject);
  check('truncated sale_listings returns no peers', empty.length === 0, `${empty.length}`);
  const emptyStats = computeSaleSimilarStats(empty, subject);
  check('empty result -> peer_count 0', emptyStats.peer_count === 0);
  check('empty result -> your_percentile defaults to 50', emptyStats.your_percentile === 50, `${emptyStats.your_percentile}`);
  check('empty result -> avg_ppsf null', emptyStats.avg_ppsf === null, `${emptyStats.avg_ppsf}`);
  check('empty result -> avg_price null', emptyStats.avg_price === null, `${emptyStats.avg_price}`);
}

async function main() {
  // Structural parity ALWAYS runs (no DB).
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
    console.log(`\n=== FAIL: ${failures} /api/similar-sale query check(s) failed. ===`);
    process.exit(1);
  }
  console.log('\n=== PASS: /api/similar-sale query behaves correctly against real Postgres ' +
    '(incl. SSTC-exclude + graceful-empty). ===');
}

main().catch((e) => {
  console.error('similar_sale_query_test crashed:', e);
  process.exit(1);
});
