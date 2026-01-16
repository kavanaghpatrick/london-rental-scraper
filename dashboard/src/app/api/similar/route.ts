import { NextRequest, NextResponse } from 'next/server';
import { getSimilarListings } from '@/lib/db';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Enable CORS for Chrome extension
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

    // Required params
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

    // Extract postcode district using regex (handles "SW3 4AJ" and "SW34AJ" formats)
    // Matches "SW1A" from "SW1A 1AA", "W1" from "W1 4BB", "EC1" from "EC1A1BB"
    const postcodeMatch = postcode.match(/^([A-Z]{1,2}[0-9][0-9A-Z]?)/i);
    const postcodeDistrict = postcodeMatch
      ? postcodeMatch[1].toUpperCase()
      : postcode.split(' ')[0].toUpperCase();

    const result = await getSimilarListings({
      postcodeDistrict,
      bedrooms: bedsNum,
      pricePcm: priceNum,
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
    console.error('Error fetching similar listings:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500, headers: corsHeaders }
    );
  }
}
