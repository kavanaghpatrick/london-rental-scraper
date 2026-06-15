import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

// CORS for the Chrome extension. Mirrors /api/similar but adds POST (the
// extension POSTs a property's raw fields and gets back a v20 fair-value).
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

export async function OPTIONS() {
  return NextResponse.json({}, { headers: corsHeaders });
}

/**
 * Request contract — the RAW property fields the extension extracts for a
 * listing. This is the input set consumed by the canonical v20 feature
 * engineering (server-side). It mirrors the input shape of the extension's
 * buildFeatures(), so the same fields drive client (fallback) and server
 * (authoritative). Finalized with artifacts (#28) / modeler (#30).
 *
 * AUTHORITATIVE estimate (ruling B): this route runs artifacts' CERTIFIED shared
 * JS predictor inline (pure JS, no Python). Modeler's corrected v20 inference
 * (#30) defines what that JS must compute and provides the golden fixtures that
 * certify it; the route serves the module's output. The same JS is the
 * extension's in-browser path, so client and server agree by construction.
 * (Both the old JS and old Python predict_one had single-row inference bugs;
 * #30's fixed output — not any earlier number — is the source of truth.)
 * See predictFairValue() below.
 */
/**
 * Lead-confirmed RAW input contract (artifacts #28). The route does NOT
 * hand-build features — it forwards these to the shared module's buildFeatures().
 * Required core drives a meaningful estimate; everything else is optional and
 * buildFeatures applies its own defaults when omitted. Unknown extra keys are
 * forwarded harmlessly.
 */
export interface PredictRequest {
  // Required core
  bedrooms: number;
  bathrooms: number;
  size_sqft: number;
  postcode: string;
  property_type: string;
  // Optional location / classification
  postcode_normalized?: string;
  area?: string;
  property_type_std?: string;
  address?: string;
  latitude?: number;
  longitude?: number;
  let_type?: string;
  // Optional text / provenance
  features?: string | string[];
  description?: string;
  source?: string;
  agent_brand?: string;
  // Optional floor flags (0/1, or floor_count integer)
  floor_count?: number;
  has_roof_terrace?: number;
  has_basement?: number;
  has_ground?: number;
  has_first_floor?: number;
  has_second_floor?: number;
  has_third_floor?: number;
  has_fourth_plus?: number;
  // Allow forward-compatible extra raw fields (forwarded to buildFeatures).
  [key: string]: unknown;
}

export interface PredictResponse {
  estimate_pcm: number;
  model_version: 'v20';
  currency: 'GBP';
  range_low: number;
  range_high: number;
}

// ---------------------------------------------------------------------------
// v20 model loading (ruling B: pure-Node route running artifacts' certified
// shared JS predictor). model.json (~11MB) + features.json are fetched from the
// GitHub raw URL the extension already uses, and cached at MODULE SCOPE so a
// warm Lambda parses once and reuses across invocations.
//
// NOTE: GitHub raw `main` only serves the v20 artifacts AFTER the unified
// go-live push. Pre-push this fetch 404s — which is fine: the route returns 503
// and the extension uses its in-browser fallback until we deploy.
// ---------------------------------------------------------------------------
const GH_RAW_BASE =
  'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api';
const MODEL_URL = `${GH_RAW_BASE}/model.json`;
const FEATURES_URL = `${GH_RAW_BASE}/features.json`;

// Thrown when the predictor backend isn't connected yet (module not wired green,
// or model not yet reachable pre-go-live). The POST handler maps it to HTTP 503
// so the extension uses its in-browser fallback.
class PredictorNotReadyError extends Error {}

// Validate the REQUIRED core fields; forward the FULL raw body (with normalized
// core values) to buildFeatures — the route is a thin pass-through, so any
// optional/extra raw fields (postcode_normalized, area, features, floor flags,
// etc.) flow straight to the shared module without being dropped.
function validate(body: unknown): { ok: true; req: PredictRequest } | { ok: false; error: string } {
  if (typeof body !== 'object' || body === null) {
    return { ok: false, error: 'Request body must be a JSON object' };
  }
  const b = body as Record<string, unknown>;

  const beds = Number(b.bedrooms);
  const baths = Number(b.bathrooms);
  const sqft = Number(b.size_sqft);
  const postcode = typeof b.postcode === 'string' ? b.postcode.trim() : '';
  const propertyType = typeof b.property_type === 'string' ? b.property_type.trim() : '';

  if (!Number.isFinite(beds) || beds < 0) {
    return { ok: false, error: 'bedrooms must be a non-negative number' };
  }
  if (!Number.isFinite(baths) || baths < 0) {
    return { ok: false, error: 'bathrooms must be a non-negative number' };
  }
  if (!Number.isFinite(sqft) || sqft <= 0) {
    return { ok: false, error: 'size_sqft must be a positive number' };
  }
  if (!postcode) {
    return { ok: false, error: 'postcode is required' };
  }
  if (!propertyType) {
    return { ok: false, error: 'property_type is required' };
  }

  return {
    ok: true,
    // Spread the raw body first (forwards all optional/extra fields untouched),
    // then overwrite the required core with normalized values.
    req: {
      ...b,
      bedrooms: beds,
      bathrooms: baths,
      size_sqft: sqft,
      postcode,
      property_type: propertyType,
    },
  };
}

/**
 * Artifacts' shared predictor module (chrome-extension/xgboost.js — ruling B).
 * VERIFIED interface (require() under Node returns these):
 *   - class XGBoostPredictor: `await p.load(modelUrl, featuresUrl)` (self-fetches +
 *     caches the parsed model at module scope via globalThis.__XGB_MODEL_CACHE__),
 *     then `p.predict(featureDict)` -> LOG prediction.
 *   - XGBFeatures.buildFeatures(rawFields) -> feature dict.
 * The module does its OWN model loading/caching, so the route does not fetch the
 * model itself — it just calls load() once (warm reuse is automatic).
 */
interface SharedPredictorModule {
  XGBoostPredictor: new () => {
    load(modelUrl: string, featuresUrl: string): Promise<void>;
    predict(featureDict: Record<string, number>): number;
  };
  XGBFeatures: { buildFeatures(raw: Record<string, unknown>): Record<string, number> };
}

/**
 * WIRED (ruling B, green 2026-06-15): artifacts' certified xgboost.js, vendored
 * into this route dir (`xgboost.predictor.js`, byte-identical copy of
 * chrome-extension/xgboost.js) so Next bundles it into the serverless function
 * (the canonical source lives outside the dashboard build root). The module is
 * GREEN: `_fixture_diff.mjs` passes 0 feature + 0 £ mismatch on all 5 locked
 * golden-fixture samples (belgravia £11,189, chelsea £4,862, mews £6,758,
 * studio £1,947, prestige £27,635).
 *
 * SYNC NOTE: if artifacts changes chrome-extension/xgboost.js, re-copy it here.
 * Parity is enforced by the golden fixture — keep this in sync at go-live.
 */
/* eslint-disable @typescript-eslint/no-require-imports */
// eslint-disable-next-line @typescript-eslint/no-var-requires
const PREDICTOR_MODULE = require('./xgboost.predictor.js') as SharedPredictorModule;

// One shared predictor instance per warm Lambda (the module also caches the
// parsed model internally, so this is belt-and-suspenders).
let predictorInstance: InstanceType<SharedPredictorModule['XGBoostPredictor']> | null = null;

/**
 * Authoritative v20 estimate (ruling B: artifacts' certified shared JS module
 * run INLINE — pure JS, no Python).
 * Flow: buildFeatures(raw) -> predict() -> expm1 -> round; range x0.79/x1.21
 * mirrors the extension (content.js) so client + server agree.
 *
 * If the model can't be loaded (e.g. GH-raw 404 pre-go-live), throws
 * PredictorNotReadyError -> HTTP 503 -> extension uses its in-browser fallback.
 */
async function predictFairValue(req: PredictRequest): Promise<PredictResponse> {
  if (!predictorInstance) {
    try {
      const p = new PREDICTOR_MODULE.XGBoostPredictor();
      // load() self-fetches model.json/features.json from GH raw and caches them.
      await p.load(MODEL_URL, FEATURES_URL);
      predictorInstance = p;
    } catch (e) {
      throw new PredictorNotReadyError(
        `v20 model not loadable yet (likely pre-go-live GH-raw 404): ${e instanceof Error ? e.message : e}`
      );
    }
  }
  const featureDict = PREDICTOR_MODULE.XGBFeatures.buildFeatures(req as Record<string, unknown>);
  const predLog = predictorInstance.predict(featureDict);
  const estimate = Math.round(Math.expm1(predLog));
  return {
    estimate_pcm: estimate,
    model_version: 'v20',
    currency: 'GBP',
    range_low: Math.round(estimate * 0.79),
    range_high: Math.round(estimate * 1.21),
  };
}

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON body' },
      { status: 400, headers: corsHeaders }
    );
  }

  const v = validate(body);
  if (!v.ok) {
    return NextResponse.json({ error: v.error }, { status: 400, headers: corsHeaders });
  }

  try {
    const result = await predictFairValue(v.req);
    return NextResponse.json(result, { headers: corsHeaders });
  } catch (error) {
    if (error instanceof PredictorNotReadyError) {
      // 503: the contract is live but the model backend is not connected yet.
      // The extension treats a non-200 as its signal to use the in-browser fallback.
      return NextResponse.json(
        { error: error.message, model_version: 'v20' },
        { status: 503, headers: corsHeaders }
      );
    }
    console.error('Error in /api/predict:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500, headers: corsHeaders }
    );
  }
}
