import { NextResponse } from 'next/server';
import { getRunningSpiders } from '@/lib/db';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    const running = await getRunningSpiders();
    return NextResponse.json(running);
  } catch (error) {
    console.error('Error fetching running spiders:', error);
    return NextResponse.json([], { status: 500 });
  }
}
