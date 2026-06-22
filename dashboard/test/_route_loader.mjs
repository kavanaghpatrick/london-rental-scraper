/**
 * Shared helper for the A6 route-HANDLER harnesses (Wave 2, Group SERVING).
 *
 * Loads an ACTUAL Next.js App-Router `route.ts` and returns its compiled exports
 * (OPTIONS / GET / POST — the real handlers), so the harnesses invoke the SAME
 * functions Vercel runs — not a re-implementation.
 *
 * HOW (no Next build, no network, no Postgres needed for the load itself):
 *   1. Transpile route.ts -> CommonJS with `sucrase` (a dependency-free TS->JS
 *      transform already present in node_modules via tailwind; transforms
 *      ["typescript","imports"]). This is the "transpile mechanism" the spec asks for.
 *   2. Compile the CJS into a fresh `node:module` Module whose `require` resolves
 *      relative to the route's own directory, so the route's own
 *      `require('./xgboost.predictor.js')` (a real committed file) loads for real.
 *   3. `next/server` (NextRequest / NextResponse) works natively under Node 18+ —
 *      `NextResponse.json()` yields a real Response with `.status`, `.headers.get()`,
 *      and `.json()`, and `NextRequest` is constructable with a body. So the handlers
 *      run end-to-end and we assert on the real HTTP envelope.
 *
 * The `requireOverrides` map lets a harness intercept a bare import the route makes —
 * specifically `@/lib/db` / `@/lib/saleDb` (which pull in @vercel/postgres and a live
 * DB). We substitute a controllable stub so we can exercise the route's 500-vs-empty
 * branch WITHOUT a database and WITHOUT editing dashboard/src. The predict / predict-sale
 * routes need no override (no DB import) — their model load is steered via the predictor's
 * model-cache global instead.
 *
 * This file is a HELPER, not a standalone `node …` harness (it exports a function and is
 * listed in dashboard_routes_guard's NOT_STANDALONE set).
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import Module from 'node:module';
import { dirname } from 'node:path';

const require = createRequire(import.meta.url);
const sucrase = require('sucrase');

/**
 * Transpile + load a route.ts, returning its handler exports.
 * @param {string} routePath absolute path to a route.ts
 * @param {Record<string, unknown>} [requireOverrides] bare-specifier -> module exports
 *        to inject (e.g. { '@/lib/db': { getSimilarListings: async () => ({...}) } }).
 * @returns {Record<string, Function>} the compiled route module exports (OPTIONS/GET/POST/…)
 */
export function loadRoute(routePath, requireOverrides = {}) {
  const src = readFileSync(routePath, 'utf8');
  const { code } = sucrase.transform(src, {
    transforms: ['typescript', 'imports'],
    filePath: routePath,
  });

  const overrideKeys = Object.keys(requireOverrides);
  const m = new Module(routePath, null);
  m.filename = routePath;
  m.paths = Module._nodeModulePaths(dirname(routePath));

  if (overrideKeys.length) {
    const origLoad = Module._load;
    Module._load = function patchedLoad(request, parent, isMain) {
      if (Object.prototype.hasOwnProperty.call(requireOverrides, request)) {
        return requireOverrides[request];
      }
      return origLoad.apply(this, arguments);
    };
    try {
      m._compile(code, routePath);
    } finally {
      Module._load = origLoad;
    }
  } else {
    m._compile(code, routePath);
  }

  return m.exports;
}

/** Build a NextRequest for a GET with a query string. */
export function getRequest(url) {
  const { NextRequest } = require('next/server');
  return new NextRequest(url, { method: 'GET' });
}

/** Build a NextRequest for a POST with a raw (possibly malformed) string body. */
export function postRequest(url, rawBody) {
  const { NextRequest } = require('next/server');
  return new NextRequest(url, {
    method: 'POST',
    body: rawBody,
    headers: { 'content-type': 'application/json' },
  });
}
