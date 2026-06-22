import { NextRequest, NextResponse } from 'next/server';
import { getSimilarSaleListings } from '@/lib/saleDb';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Enable CORS for Chrome extension. Mirrors /api/similar (GET) — the FOR-SALE analogue.
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

export async function OPTIONS() {
  return NextResponse.json({}, { headers: corsHeaders });
}

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    const { searchParams } = new URL(request.url);

    // Required params (price = ASKING price for sale)
    const postcode = searchParams.get('postcode');
    const beds = searchParams.get('beds');
    const price = searchParams.get('price');

    if (!postcode || !beds || !price) {
      return NextResponse.json(
        { error: 'Missing required params: postcode, beds, price' },
        { status: 400, headers: corsHeaders }
      );
    }

    // Parse and validate numeric params
    const bedsNum = parseInt(beds, 10);
    const priceNum = parseInt(price, 10);

    if (isNaN(bedsNum) || bedsNum < 0) {
      return NextResponse.json(
        { error: 'Invalid beds parameter: must be a non-negative integer' },
        { status: 400, headers: corsHeaders }
      );
    }

    if (isNaN(priceNum) || priceNum <= 0) {
      return NextResponse.json(
        { error: 'Invalid price parameter: must be a positive integer' },
        { status: 400, headers: corsHeaders }
      );
    }

    // Optional params
    const sqft = searchParams.get('sqft');
    const type = searchParams.get('type');
    const excludeId = searchParams.get('exclude');

    // Validate optional sqft if provided
    let sqftNum: number | undefined;
    if (sqft) {
      const parsed = parseInt(sqft, 10);
      if (!isNaN(parsed) && parsed > 0) {
        sqftNum = parsed;
      }
    }

    // Extract postcode district using regex (handles "SW3 4AJ" and "SW34AJ" formats).
    // The lookahead `(?=\s|[0-9]|$)` stops the outward code BEFORE the incode digit so a
    // no-space 'SW34AJ' yields 'SW3' (NOT the greedy 'SW34'). Mirrors the Python _PC_RE.
    const postcodeMatch = postcode.match(/^([A-Z]{1,2}[0-9][0-9A-Z]?)(?=\s|[0-9]|$)/i);
    const postcodeDistrict = postcodeMatch
      ? postcodeMatch[1].toUpperCase()
      : postcode.split(' ')[0].toUpperCase();

    // A MISSING sale_listings table is NOT an error — getSimilarSaleListings degrades to
    // empty peers (graceful-empty, AMENDMENT FIX 6). Only a GENUINE failure hits the catch.
    const result = await getSimilarSaleListings({
      postcodeDistrict,
      bedrooms: bedsNum,
      askingPrice: priceNum,
      sizeSqft: sqftNum,
      propertyType: type || undefined,
      excludeId: excludeId || undefined,
    });

    const queryMs = Date.now() - startTime;

    return NextResponse.json(
      { ...result, query_ms: queryMs },
      { headers: corsHeaders }
    );
  } catch (error) {
    console.error('Error fetching similar sale listings:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500, headers: corsHeaders }
    );
  }
}
