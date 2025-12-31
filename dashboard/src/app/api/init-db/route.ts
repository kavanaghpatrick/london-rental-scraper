import { sql } from '@vercel/postgres';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Create listings table
    await sql`
      CREATE TABLE IF NOT EXISTS listings (
        id SERIAL PRIMARY KEY,
        source TEXT NOT NULL,
        property_id TEXT NOT NULL,
        url TEXT,
        area TEXT,
        price INTEGER,
        price_pw INTEGER,
        price_pcm INTEGER,
        price_period TEXT,
        address TEXT,
        postcode TEXT,
        latitude REAL,
        longitude REAL,
        bedrooms INTEGER,
        bathrooms INTEGER,
        reception_rooms INTEGER,
        property_type TEXT,
        size_sqft INTEGER,
        size_sqm REAL,
        furnished TEXT,
        epc_rating TEXT,
        floorplan_url TEXT,
        room_details TEXT,
        has_basement INTEGER DEFAULT 0,
        has_lower_ground INTEGER DEFAULT 0,
        has_ground INTEGER DEFAULT 0,
        has_mezzanine INTEGER DEFAULT 0,
        has_first_floor INTEGER DEFAULT 0,
        has_second_floor INTEGER DEFAULT 0,
        has_third_floor INTEGER DEFAULT 0,
        has_fourth_plus INTEGER DEFAULT 0,
        has_roof_terrace INTEGER DEFAULT 0,
        floor_count INTEGER,
        property_levels TEXT,
        let_agreed INTEGER DEFAULT 0,
        agent_name TEXT,
        agent_phone TEXT,
        summary TEXT,
        description TEXT,
        features TEXT,
        added_date TEXT,
        address_fingerprint TEXT,
        first_seen TEXT,
        last_seen TEXT,
        is_active INTEGER DEFAULT 1,
        price_change_count INTEGER DEFAULT 0,
        scraped_at TEXT,
        -- V15 model columns
        is_short_let INTEGER DEFAULT 0,
        property_type_std TEXT,
        let_type TEXT,
        postcode_normalized TEXT,
        postcode_inferred TEXT,
        agent_brand TEXT,
        UNIQUE(source, property_id)
      )
    `;

    // Create price_history table
    await sql`
      CREATE TABLE IF NOT EXISTS price_history (
        id SERIAL PRIMARY KEY,
        listing_id INTEGER NOT NULL REFERENCES listings(id),
        price_pcm INTEGER,
        recorded_at TEXT
      )
    `;

    // Create scrape_runs table
    await sql`
      CREATE TABLE IF NOT EXISTS scrape_runs (
        id SERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        spider_name TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        duration_seconds REAL,
        status TEXT DEFAULT 'running',
        items_scraped INTEGER DEFAULT 0,
        items_new INTEGER DEFAULT 0,
        items_updated INTEGER DEFAULT 0,
        items_dropped INTEGER DEFAULT 0,
        items_errors INTEGER DEFAULT 0,
        request_count INTEGER DEFAULT 0,
        response_count INTEGER DEFAULT 0,
        response_bytes BIGINT DEFAULT 0,
        error_count INTEGER DEFAULT 0,
        retry_count INTEGER DEFAULT 0,
        memory_start_mb REAL,
        memory_peak_mb REAL,
        memory_end_mb REAL,
        log_file TEXT,
        exit_reason TEXT,
        error_summary TEXT,
        UNIQUE(run_id, spider_name)
      )
    `;

    // Create scrape_events table
    await sql`
      CREATE TABLE IF NOT EXISTS scrape_events (
        id SERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        spider_name TEXT NOT NULL,
        event_type TEXT NOT NULL,
        event_time TEXT NOT NULL,
        message TEXT,
        details TEXT,
        severity TEXT DEFAULT 'info'
      )
    `;

    // Create model_runs table (for V15 model training metrics)
    await sql`
      CREATE TABLE IF NOT EXISTS model_runs (
        id SERIAL PRIMARY KEY,
        run_date TEXT NOT NULL,
        run_id TEXT NOT NULL,
        version TEXT NOT NULL,
        samples_total INTEGER,
        features_count INTEGER,
        r2_score REAL,
        mae REAL,
        mape REAL,
        median_ape REAL,
        training_time_seconds REAL,
        best_params TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `;

    // Create property_valuations table (for storing predictions)
    await sql`
      CREATE TABLE IF NOT EXISTS property_valuations (
        id SERIAL PRIMARY KEY,
        address TEXT NOT NULL,
        postcode TEXT,
        size_sqft INTEGER,
        bedrooms INTEGER,
        bathrooms INTEGER,
        predicted_pcm REAL,
        range_low REAL,
        range_high REAL,
        model_version TEXT,
        model_r2 REAL,
        model_mape REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `;

    // Add V15 model columns to existing tables (migrations)
    const v15Columns = [
      { name: 'is_short_let', type: 'INTEGER DEFAULT 0' },
      { name: 'property_type_std', type: 'TEXT' },
      { name: 'let_type', type: 'TEXT' },
      { name: 'postcode_normalized', type: 'TEXT' },
      { name: 'postcode_inferred', type: 'TEXT' },
      { name: 'agent_brand', type: 'TEXT' },
    ];

    for (const col of v15Columns) {
      try {
        await sql.query(`ALTER TABLE listings ADD COLUMN IF NOT EXISTS ${col.name} ${col.type}`);
      } catch (e) {
        // Column might already exist, ignore
      }
    }

    // Create indexes
    await sql`CREATE INDEX IF NOT EXISTS idx_listings_source_prop ON listings(source, property_id)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_listings_fingerprint ON listings(address_fingerprint)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_listings_active ON listings(is_active)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_price_history_listing ON price_history(listing_id)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_runs_run_id ON scrape_runs(run_id)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_runs_started ON scrape_runs(started_at)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_events_run_id ON scrape_events(run_id)`;

    return NextResponse.json({ 
      success: true, 
      message: 'Database schema initialized successfully' 
    });
  } catch (error) {
    console.error('Database initialization error:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      },
      { status: 500 }
    );
  }
}
