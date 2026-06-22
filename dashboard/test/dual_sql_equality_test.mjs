/**
 * A7 — DUAL-SQL byte-equality (Wave 2, Group SERVING).
 *
 * /api/similar runs an INLINE tagged-template SQL in db.ts getSimilarListings. The CI
 * harness (similar_query_test.mjs) instead runs buildSimilarQuery() from similarQuery.js
 * against a Postgres service container. similar_query_test.mjs already checks that a HANDFUL
 * of load-bearing TOKENS appear in both — but token-presence can't catch a divergence in,
 * say, a BETWEEN bound, a CASE weight ordering, or an added/removed WHERE clause. If the two
 * SQLs silently drift, prod (db.ts) ships UNTESTED logic while the green test exercises a
 * different query.
 *
 * THIS test closes that gap with a FULL byte-equality assertion: it extracts the inline SQL
 * body from db.ts, rewrites its `${jsVar}` tagged-template interpolations to the SAME
 * positional `$N` placeholders buildSimilarQuery() emits (using the var->position map the
 * module's `values` array defines), strips `--` comments, collapses whitespace, and asserts
 * the two normalized strings are IDENTICAL. Same for saleDb.ts vs saleSimilarQuery.js.
 *
 * If anyone edits the inline prod SQL without mirroring it into the tested module (or vice
 * versa), the normalized strings diverge and this FAILS — so the tested copy can never
 * silently diverge from the deployed query.
 *
 * Pure text, no DB, no network. Run: node dashboard/test/dual_sql_equality_test.mjs
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const LIB = join(__dirname, '..', 'src', 'lib');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

/**
 * Extract the `WITH scored AS ( … LIMIT 15` SQL body of a named query function from a
 * db.ts / saleDb.ts source. Anchored on the function name so we slice the RIGHT query
 * (these files contain many tagged-template queries).
 */
function extractInlineSql(srcFile, anchorFn) {
  const src = readFileSync(join(LIB, srcFile), 'utf8');
  const fnIdx = src.indexOf(anchorFn);
  if (fnIdx < 0) throw new Error(`anchor ${anchorFn} not found in ${srcFile}`);
  const start = src.indexOf('WITH scored AS', fnIdx);
  if (start < 0) throw new Error(`'WITH scored AS' not found after ${anchorFn} in ${srcFile}`);
  const limitTok = 'LIMIT 15';
  const end = src.indexOf(limitTok, start);
  if (end < 0) throw new Error(`'LIMIT 15' not found after ${anchorFn} in ${srcFile}`);
  return src.slice(start, end + limitTok.length);
}

/**
 * Normalize an SQL body to a canonical comparable form:
 *   - rewrite ${jsVar} -> $N using varMap (db.ts side only; the module already uses $N)
 *   - strip `--` line comments
 *   - collapse all whitespace to single spaces, trim
 * Any ${var} not present in varMap throws (forces the map to stay complete — a new
 * interpolation in prod that the map doesn't know about is a loud failure, not a silent pass).
 */
function normalizeSql(sql, varMap) {
  let s = sql.replace(/--[^\n]*/g, '');
  if (varMap) {
    s = s.replace(/\$\{\s*([A-Za-z0-9_]+)\s*\}/g, (_m, name) => {
      if (!(name in varMap)) throw new Error(`UNMAPPED interpolation \${${name}} — extend the var->position map`);
      return varMap[name];
    });
  }
  return s.replace(/\s+/g, ' ').trim();
}

/**
 * The var->position map for the RENTAL inline query. MUST match the order of the `values`
 * array in similarQuery.js buildSimilarQuery() ($1..$16). If similarQuery.js reorders its
 * params, this map must follow — and the byte-equality below is what enforces it.
 */
const RENTAL_VAR_MAP = {
  safeBedrooms: '$1',
  safeSizeSqft: '$2',
  minSqft: '$3',
  maxSqft: '$4',
  minSqftWide: '$5',
  maxSqftWide: '$6',
  safePricePcm: '$7',
  priceTolerance15: '$8',
  priceTolerance30: '$9',
  safePropertyType: '$10',
  postcodeDistrict: '$11',
  minBedrooms: '$12',
  maxBedrooms: '$13',
  priceRangeMin: '$14',
  priceRangeMax: '$15',
  safeExcludeId: '$16',
};

/** Sale analogue — same positions, but the price var is safeAskingPrice (sale column). */
const SALE_VAR_MAP = {
  safeBedrooms: '$1',
  safeSizeSqft: '$2',
  minSqft: '$3',
  maxSqft: '$4',
  minSqftWide: '$5',
  maxSqftWide: '$6',
  safeAskingPrice: '$7',
  priceTolerance15: '$8',
  priceTolerance30: '$9',
  safePropertyType: '$10',
  postcodeDistrict: '$11',
  minBedrooms: '$12',
  maxBedrooms: '$13',
  priceRangeMin: '$14',
  priceRangeMax: '$15',
  safeExcludeId: '$16',
};

function firstDiff(a, b) {
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] !== b[i]) {
      return `@${i}: inline=${JSON.stringify(a.slice(Math.max(0, i - 40), i + 40))} module=${JSON.stringify(b.slice(Math.max(0, i - 40), i + 40))}`;
    }
  }
  return `lengths differ: inline=${a.length} module=${b.length}`;
}

function main() {
  // -------------------- RENTAL: db.ts vs similarQuery.js --------------------
  {
    const { buildSimilarQuery } = require(join(LIB, 'similarQuery.js'));
    const inline = normalizeSql(extractInlineSql('db.ts', 'getSimilarListings'), RENTAL_VAR_MAP);
    const moduleSql = normalizeSql(
      buildSimilarQuery({
        postcodeDistrict: 'SW3', bedrooms: 2, pricePcm: 5000, sizeSqft: 1000, propertyType: 'flat',
      }).text,
      null
    );
    check('RENTAL inline SQL is non-trivial (sanity)', inline.length > 400, `${inline.length} chars`);
    check(
      'RENTAL /api/similar: db.ts inline SQL == similarQuery.js buildSimilarQuery (normalized byte-equal)',
      inline === moduleSql,
      inline === moduleSql ? '' : firstDiff(inline, moduleSql)
    );
  }

  // -------------------- SALE: saleDb.ts vs saleSimilarQuery.js --------------------
  {
    const { buildSaleSimilarQuery } = require(join(LIB, 'saleSimilarQuery.js'));
    const inline = normalizeSql(extractInlineSql('saleDb.ts', 'getSimilarSaleListings'), SALE_VAR_MAP);
    const moduleSql = normalizeSql(
      buildSaleSimilarQuery({
        postcodeDistrict: 'SW3', bedrooms: 2, askingPrice: 3_000_000, sizeSqft: 1000, propertyType: 'flat',
      }).text,
      null
    );
    check('SALE inline SQL is non-trivial (sanity)', inline.length > 400, `${inline.length} chars`);
    check(
      'SALE /api/similar-sale: saleDb.ts inline SQL == saleSimilarQuery.js buildSaleSimilarQuery (normalized byte-equal)',
      inline === moduleSql,
      inline === moduleSql ? '' : firstDiff(inline, moduleSql)
    );
    // Sale-specific guard: the under-offer (SSTC) exclusion must be in BOTH copies — the
    // single clause that distinguishes the sale query from a rename of the rental one.
    check('SALE both copies contain the is_under_offer (SSTC) exclusion',
      inline.includes('is_under_offer') && moduleSql.includes('is_under_offer'));
  }

  if (failures) {
    console.log(`\n=== FAIL: ${failures} dual-SQL equality check(s) failed — prod inline SQL and the tested module have DIVERGED. ===`);
    process.exit(1);
  }
  console.log('\n=== PASS: inline route SQL is byte-equal to the tested query module (rental + sale). ===');
}

try {
  main();
} catch (e) {
  console.error('dual_sql_equality_test crashed:', e);
  process.exit(1);
}
