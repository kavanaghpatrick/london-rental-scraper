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

    // Optional params
    const sqft = searchParams.get('sqft');
    const type = searchParams.get('type');
    const excludeId = searchParams.get('exclude');

    // Extract postcode district (SW3, W8, NW1, etc.)
    const postcodeDistrict = postcode.split(' ')[0].toUpperCase();

    const result = await getSimilarListings({
      postcodeDistrict,
      bedrooms: parseInt(beds, 10),
      pricePcm: parseInt(price, 10),
      sizeSqft: sqft ? parseInt(sqft, 10) : undefined,
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
