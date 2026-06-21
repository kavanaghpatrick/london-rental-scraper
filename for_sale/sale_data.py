"""for_sale.sale_data — the ISOLATED data layer for the FOR-SALE vertical.

This module persists SCRAPED FOR-SALE ASKING listings (from the spiders' for-sale mode)
into their OWN `sale_listings` table, with the asking price in a DISTINCT `asking_price`
column. It is deliberately separate from the rental `listings` table so a sale price
(a £100k–£50M lump sum) can never leak into the rental price_pcm magnitude and vice-versa.

ISOLATION CONTRACT
------------------
  * Never imports the parity-gated rental MODEL chain (rental_price_models_v20 /
    canonical_predict). A rental retrain can never perturb the for-sale layer.
  * Never CREATEs or writes the rental `listings` table.
  * It MAY reuse SHARED SCRAPING INFRA (the address-fingerprint primitive) — that is a
    pattern, not the rental model — so the two verticals agree on a property's identity
    hash without the for-sale layer copy-pasting the algorithm.

It mirrors the rental pipeline's identity rule (UNIQUE(source, property_id)) and its
upsert-with-fingerprint shape, but on the sale schema. Keep this pure and DB-handle
based (caller supplies the sqlite3 connection) so it is trivially unit-testable in CI
with an in-memory DB — no live file, no network.
"""
from __future__ import annotations

import sqlite3
from typing import Any

# SHARED scraping infra (NOT the rental model) — reused so both verticals derive the
# same address_fingerprint for a given address. Banning this would force a copy-paste of
# the fingerprint algorithm, the exact divergence the rental dedupe fix warns against.
from property_scraper.services.fingerprint import generate_fingerprint

# The for-sale table's column set. Sale magnitude lives in `asking_price`; there is NO
# price_pcm/price_pw column here by construction. `price_qualifier` (Guide Price / Offers
# in Excess of / POA / '') is sale-specific and has no rental analogue.
SALE_COLUMNS: tuple[str, ...] = (
    "source",
    "property_id",
    "url",
    "area",
    "asking_price",
    "price_qualifier",
    "address",
    "postcode",
    "latitude",
    "longitude",
    "bedrooms",
    "bathrooms",
    "property_type",
    "size_sqft",
    # Sale-specific status (no rental analogue): a buyer-side tool must be able to
    # EXCLUDE under-offer/SSTC comps and flag new-builds (a known price-magnitude
    # outlier the Inc3 sale model treats specially). The parser already extracts these;
    # they MUST be persisted, not dropped.
    "is_new_build",
    "is_under_offer",
    "agent_name",
    "agent_phone",
    "summary",
    "added_date",
    "address_fingerprint",
    "first_seen",
    "last_seen",
    "is_active",
    "scraped_at",
)


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the isolated `sale_listings` table (idempotent).

    Mirrors the rental pipeline's identity (UNIQUE(source, property_id)) but on the sale
    schema. Does NOT touch the rental `listings` table.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sale_listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            property_id TEXT NOT NULL,
            url TEXT,
            area TEXT,
            asking_price INTEGER,           -- ASKING SALE PRICE (lump sum); NULL = POA
            price_qualifier TEXT,           -- Guide Price / Offers in Excess of / POA / ''
            address TEXT,
            postcode TEXT,
            latitude REAL,
            longitude REAL,
            bedrooms INTEGER,
            bathrooms INTEGER,
            property_type TEXT,
            size_sqft INTEGER,
            is_new_build INTEGER DEFAULT 0,   -- 1 = new-home/development listing
            is_under_offer INTEGER DEFAULT 0, -- 1 = SSTC / under offer (excludable comp)
            agent_name TEXT,
            agent_phone TEXT,
            summary TEXT,
            added_date TEXT,
            address_fingerprint TEXT,
            first_seen TEXT,
            last_seen TEXT,
            is_active INTEGER DEFAULT 1,
            scraped_at TEXT,
            UNIQUE(source, property_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sale_source_prop "
        "ON sale_listings(source, property_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sale_fingerprint "
        "ON sale_listings(address_fingerprint)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sale_postcode ON sale_listings(postcode)"
    )
    conn.commit()


def _fingerprint(item: dict[str, Any]) -> str | None:
    """Derive the address fingerprint via the SHARED primitive (best-effort)."""
    address = item.get("address") or ""
    postcode = item.get("postcode") or ""
    if not address:
        return None
    try:
        return generate_fingerprint(address, postcode)
    except Exception:
        # Fingerprint is an optimisation for cross-source dedupe, not a correctness
        # requirement for persistence — never let it block an upsert.
        return None


def upsert_sale_listing(conn: sqlite3.Connection, item: dict[str, Any]) -> None:
    """Insert or update one for-sale listing keyed on (source, property_id).

    Only the SALE_COLUMNS are persisted — any stray key (e.g. a rental `price_pcm`) is
    DROPPED, never written, so the rental magnitude can never enter the sale table. The
    address_fingerprint is derived here if the caller didn't supply one.
    """
    row = {k: item.get(k) for k in SALE_COLUMNS}

    if not row.get("address_fingerprint"):
        row["address_fingerprint"] = _fingerprint(item)

    cols = ", ".join(SALE_COLUMNS)
    placeholders = ", ".join("?" for _ in SALE_COLUMNS)
    # On conflict (same source+property_id) update the mutable fields in place — no dup.
    updatable = [c for c in SALE_COLUMNS if c not in ("source", "property_id", "first_seen")]
    update_clause = ", ".join(f"{c}=excluded.{c}" for c in updatable)

    conn.execute(
        f"""
        INSERT INTO sale_listings ({cols})
        VALUES ({placeholders})
        ON CONFLICT(source, property_id) DO UPDATE SET {update_clause}
        """,
        [row[c] for c in SALE_COLUMNS],
    )
    conn.commit()


def fetch_sale_listings(
    conn: sqlite3.Connection, active_only: bool = False
) -> list[dict[str, Any]]:
    """Read sale listings back as dicts (column-name keyed), newest-id first stable order."""
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT * FROM sale_listings"
        if active_only:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY id"
        cur = conn.execute(sql)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.row_factory = prev_factory
