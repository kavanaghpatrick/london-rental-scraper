import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

// CORS for the Chrome extension. Mirrors /api/predict (POST a property's raw
// fields, get back a sale_v1 fair-value lump sum). FULLY ISOLATED from the rental
// route — separate predictor module, separate artifacts (output/sale_api/),
// separate model-cache global (__SALE_XGB_MODEL_CACHE__ inside the predictor),
// separate GH-raw base. Nothing here imports the rental route/db/predictor.
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
 * FOR-SALE listing. Forwarded straight to the shared sale module's
 * buildFeatures(), so the same fields drive client (in-browser fallback) and
 * server (authoritative). The sale predictor is the SALE analogue of the rental
 * xgboost.js, ported by value and vendored here as sale_xgboost.predictor.js.
 *
 * DIVERGENCE FROM RENTAL: postcode is NOT hard-required and size_sqft MAY be ≤0.
 * The sale model tolerates a missing postcode (-> district UNKNOWN -> low_confidence)
 * and a missing/zero size (-> 700 fallback -> estimated_size). validate() coerces
 * beds/baths to finite ≥0 and normalizes postcode/size WITHOUT 400-ing — the UX
 * signal is carried by the low_confidence / estimated_size response flags instead.
 * Only a non-object body (or invalid JSON) -> 400.
 */
export interface PredictSaleRequest {
  // Core (postcode + size_sqft are NOT hard-required — see divergence note above)
  postcode: string;
  bedrooms: number;
  bathrooms: number;
  size_sqft: number;
  property_type: string;
  // Optional location / classification
  address?: string;
  is_new_build?: number;
  latitude?: number;
  longitude?: number;
  price_qualifier?: string;
  // Allow forward-compatible extra raw fields (forwarded to buildFeatures).
  [key: string]: unknown;
}

export interface PredictSaleResponse {
  predicted_price: number;
  model_version: 'sale_v1';
  currency: 'GBP';
  range_low: number;
  range_high: number;
  low_confidence: boolean;
  district: string;
  estimated_size: boolean;
}

// ---------------------------------------------------------------------------
// Sale model loading (pure-Node route running the certified shared sale JS
// predictor). model.json + features.json are fetched from the GitHub raw URL —
// SALE-ISOLATED at output/sale_api/ (NEVER chrome-extension/api). The inference
// maps (district_freq / postcode_area_freq) are BAKED into the predictor module,
// so the route needs only model + features.
//
// NOTE: GitHub raw `main` only serves output/sale_api/* AFTER the unified
// go-live push. Pre-push this fetch 404s — which is fine: the route returns 503
// and the extension uses its in-browser fallback until we deploy.
// ---------------------------------------------------------------------------
const SALE_GH_RAW_BASE =
  'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/output/sale_api';
const MODEL_URL = `${SALE_GH_RAW_BASE}/model.json`;
const FEATURES_URL = `${SALE_GH_RAW_BASE}/features.json`;

// Sale asking-price range band — TIGHTER than the rental 0.79/1.21 (asking-price
// dispersion is narrower than achieved-rent dispersion). NEW sale-specific
// constant pair, documented here.
const SALE_RANGE_LOW_FACTOR = 0.85;
const SALE_RANGE_HIGH_FACTOR = 1.15;

// Sale predicted-price clamp (mirrors the Python predict_one_default UX layer).
const SALE_PRICE_MIN = 50000;
const SALE_PRICE_MAX = 250000000;

// Thrown when the sale predictor backend isn't connected yet (model not yet
// reachable pre-go-live). The POST handler maps it to HTTP 503 so the extension
// uses its in-browser fallback.
class PredictorNotReadyError extends Error {}

// Validate the body. UNLIKE the rental route, postcode + size_sqft do NOT 400 —
// they are normalized and the predictor's low_confidence / estimated_size flags
// carry the UX signal. Only a non-object body fails here.
function validate(
  body: unknown
): { ok: true; req: PredictSaleRequest } | { ok: false; error: string } {
  if (typeof body !== 'object' || body === null) {
    return { ok: false, error: 'Request body must be a JSON object' };
  }
  const b = body as Record<string, unknown>;

  const beds = Number(b.bedrooms);
  const baths = Number(b.bathrooms);
  // size_sqft is allowed to be ≤0 / missing — the predictor falls back to 700.
  const sqft = Number(b.size_sqft);
  // postcode is allowed to be empty — the predictor maps it to district UNKNOWN.
  const postcode = typeof b.postcode === 'string' ? b.postcode.trim() : '';
  const propertyType = typeof b.property_type === 'string' ? b.property_type.trim() : '';

  const safeBeds = Number.isFinite(beds) && beds >= 0 ? beds : 0;
  const safeBaths = Number.isFinite(baths) && baths >= 0 ? baths : 0;

  return {
    ok: true,
    // Spread the raw body first (forwards all optional/extra fields untouched),
    // then overwrite the normalized core. size_sqft is forwarded as-is (NaN/≤0
    // allowed) so the predictor applies its 700 fallback; postcode may be ''.
    req: {
      ...b,
      bedrooms: safeBeds,
      bathrooms: safeBaths,
      size_sqft: sqft,
      postcode,
      property_type: propertyType,
    },
  };
}

/**
 * The certified shared SALE predictor module (chrome-extension/sale_xgboost.js,
 * vendored here byte-identically as sale_xgboost.predictor.js so Next bundles it
 * into the serverless function). The byte-identity is enforced by the parity test.
 *
 * VERIFIED interface (require() under Node returns these):
 *   - class SaleXGBoostPredictor: `await p.load(modelUrl, featuresUrl)` (self-fetches
 *     + caches the parsed model at module scope via globalThis.__SALE_XGB_MODEL_CACHE__),
 *     then `p.predict(featureDict)` -> LOG prediction.
 *   - SaleXGBFeatures.buildFeatures(rawFields) -> 34-feature dict.
 */
interface SaleSharedPredictorModule {
  SaleXGBoostPredictor: new () => {
    load(modelUrl: string, featuresUrl: string): Promise<void>;
    predict(featureDict: Record<string, number>): number;
  };
  SaleXGBFeatures: { buildFeatures(raw: Record<string, unknown>): Record<string, number> };
}

/* eslint-disable @typescript-eslint/no-require-imports */
// eslint-disable-next-line @typescript-eslint/no-var-requires
const PREDICTOR_MODULE = require('./sale_xgboost.predictor.js') as SaleSharedPredictorModule;

// One shared sale predictor instance per warm Lambda (the module also caches the
// parsed model internally, so this is belt-and-suspenders).
let predictorInstance: InstanceType<SaleSharedPredictorModule['SaleXGBoostPredictor']> | null = null;

// Extract the postcode district exactly like the predictor does (UNKNOWN when
// absent) so the response `district` + `low_confidence` agree with the features.
function extractDistrict(postcode: string): string {
  if (!postcode) return 'UNKNOWN';
  const m = String(postcode).trim().toUpperCase().match(/^([A-Z]{1,2}\d{1,2}[A-Z]?)(?=\s|\d|$)/);
  return m ? m[1] : 'UNKNOWN';
}

/**
 * Authoritative sale_v1 estimate (certified shared sale JS module run INLINE —
 * pure JS, no Python).
 * Flow: buildFeatures(raw) -> predict() -> expm1 -> round -> clamp; range
 * x0.85/x1.15. Ports predict_one_default's UX layer (district / estimated_size /
 * low_confidence) in JS.
 *
 * If the model can't be loaded (e.g. GH-raw 404 pre-go-live), throws
 * PredictorNotReadyError -> HTTP 503 -> extension uses its in-browser fallback.
 */
async function predictSaleValue(req: PredictSaleRequest): Promise<PredictSaleResponse> {
  if (!predictorInstance) {
    try {
      const p = new PREDICTOR_MODULE.SaleXGBoostPredictor();
      // load() self-fetches model.json/features.json from GH raw and caches them.
      await p.load(MODEL_URL, FEATURES_URL);
      predictorInstance = p;
    } catch (e) {
      throw new PredictorNotReadyError(
        `sale_v1 model not loadable yet (likely pre-go-live GH-raw 404): ${e instanceof Error ? e.message : e}`
      );
    }
  }

  const featureDict = PREDICTOR_MODULE.SaleXGBFeatures.buildFeatures(req as Record<string, unknown>);
  const predLog = predictorInstance.predict(featureDict);
  const price = Math.round(Math.expm1(predLog));
  const clamped = Math.min(Math.max(price, SALE_PRICE_MIN), SALE_PRICE_MAX);

  // UX layer (mirrors predict_one_default): a missing/≤0 size means the predictor
  // used its 700 fallback (estimated_size); a missing postcode -> UNKNOWN district.
  const district = extractDistrict(req.postcode);
  const estimated_size = !(Number(req.size_sqft) > 0);
  const low_confidence = estimated_size || district === 'UNKNOWN';

  return {
    predicted_price: clamped,
    model_version: 'sale_v1',
    currency: 'GBP',
    range_low: Math.round(clamped * SALE_RANGE_LOW_FACTOR),
    range_high: Math.round(clamped * SALE_RANGE_HIGH_FACTOR),
    low_confidence,
    district,
    estimated_size,
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
    const result = await predictSaleValue(v.req);
    return NextResponse.json(result, { headers: corsHeaders });
  } catch (error) {
    if (error instanceof PredictorNotReadyError) {
      // 503: the contract is live but the model backend is not connected yet.
      // The extension treats a non-200 as its signal to use the in-browser fallback.
      return NextResponse.json(
        { error: error.message, model_version: 'sale_v1' },
        { status: 503, headers: corsHeaders }
      );
    }
    console.error('Error in /api/predict-sale:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500, headers: corsHeaders }
    );
  }
}
