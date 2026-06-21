"""build_sale_training_sample.py — generate the COMMITTED deterministic for-sale
training sample used by tests/test_for_sale_sale_model.py.

WHY THIS IS SYNTHETIC (and that is correct for THIS increment)
--------------------------------------------------------------
The captured live for-sale fixtures (rightmove_for_sale_properties.json, 6 rows;
foxtons_for_sale_properties.json, 6 rows) are real but far too few — and too sparsely
priced — to train AND validate a model with a held-out R² assertion in CI. So the
BASELINE model is exercised against a DETERMINISTIC sample generated here from a
DOCUMENTED, leak-free London sale-price formula:

    asking_price ≈ size_sqft * base_ppsf(district) * type_mult * prestige_mult
                   * (1 + bounded seeded noise)

This is the standard "can the pipeline learn a known signal" baseline check — the
formula is realistic (prime-London sale ppsf ~£700–£2,800/sqft, houses > flats,
prestige streets carry a premium) but the model never sees the formula, only the
features. Because asking_price is a function of the SAME property attributes the model
features on (NOT of any hidden target-derived quantity), a competent XGBoost clears
R² > 0.6 on the held-out split, proving the feature-engineering + training path works.

When real scraped for-sale data accumulates (Inc2 wires the spider sale-mode into
output/sales.db), the model trains on THAT instead — train() takes a list of row dicts
from any source. This sample is only the CI-deterministic stand-in, exactly analogous
to how the rental tests use small committed fixtures rather than the live DB.

Regenerate with:  python3 -m for_sale.build_sale_training_sample
It is fully deterministic (fixed seed) so re-running yields a byte-identical file.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "for_sale" / "sale_training_sample.json"

# Base sale price-per-sqft by district (£/sqft) — realistic prime/greater-London scale.
DISTRICT_PPSF = {
    "SW1X": 2_800, "SW3": 2_400, "SW7": 2_300, "SW1": 2_100, "W8": 2_200,
    "W11": 1_900, "W1K": 2_600, "SW10": 1_800, "NW8": 1_700, "NW3": 1_500,
    "SW5": 1_400, "SW6": 1_250, "SE1": 1_100, "E14": 1_000, "SE15": 750,
    "UNKNOWN": 950,
}
# Streets that carry an extra prestige premium (subset of the rental prestige list).
PRESTIGE_STREETS = {
    "cadogan square": 1.35, "eaton square": 1.40, "the boltons": 1.45,
    "cheyne walk": 1.30, "onslow square": 1.20, "sloane street": 1.25,
    "hans place": 1.30, "lennox gardens": 1.18, "tregunter road": 1.22,
}
TYPE_MULT = {"House": 1.25, "Town House": 1.22, "Penthouse": 1.30,
             "Flat": 1.0, "Apartment": 1.0, "Maisonette": 1.05}

# Address stems per district so the prestige-street signal is learnable.
DISTRICT_STREETS = {
    "SW3": ["Cadogan Square", "Draycott Place", "Tedworth Square", "Flood Street"],
    "SW1X": ["Eaton Square", "Lowndes Square", "Chesham Place"],
    "SW7": ["Onslow Square", "Queens Gate", "Cornwall Gardens"],
    "SW10": ["The Boltons", "Tregunter Road", "Gilston Road", "Cathcart Road"],
    "W8": ["Kensington Court", "Stafford Terrace", "Phillimore Gardens"],
    "W11": ["Lansdowne Road", "Elgin Crescent", "Ladbroke Grove"],
    "NW3": ["Frognal", "Belsize Park Gardens", "Fitzjohns Avenue"],
    "SE15": ["Bellenden Road", "Choumert Road", "Nunhead Lane"],
    "E14": ["Westferry Road", "Marsh Wall", "Manilla Street"],
    "SW6": ["Fulham Road", "Wandsworth Bridge Road", "Munster Road"],
}


def _district_for(i: int) -> str:
    return list(DISTRICT_STREETS.keys())[i % len(DISTRICT_STREETS)]


def build(n: int = 300, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(n):
        district = _district_for(i)
        streets = DISTRICT_STREETS[district]
        street = streets[i % len(streets)]
        beds = rng.choice([1, 1, 2, 2, 2, 3, 3, 4, 5])
        baths = max(1, beds - rng.choice([0, 0, 1]))
        # Size scales with beds, with realistic spread.
        size = int(rng.gauss(350 + beds * 320, 120))
        size = max(380, min(size, 9000))
        ptype = rng.choice(["Flat", "Flat", "Flat", "House", "Town House",
                            "Apartment", "Penthouse", "Maisonette"])

        base_ppsf = DISTRICT_PPSF.get(district, DISTRICT_PPSF["UNKNOWN"])
        type_mult = TYPE_MULT.get(ptype, 1.0)
        prestige_mult = PRESTIGE_STREETS.get(street.lower(), 1.0)
        noise = 1.0 + rng.gauss(0, 0.05)  # ±5% bounded asking-price scatter
        price = int(size * base_ppsf * type_mult * prestige_mult * noise)
        price = max(120_000, min(price, 55_000_000))

        rows.append({
            "source": "synthetic",
            "property_id": f"S{i:04d}",
            "address": f"{street}, London, {district}",
            "postcode": district if district != "UNKNOWN" else "",
            "bedrooms": beds,
            "bathrooms": baths,
            "size_sqft": size,
            "property_type": ptype,
            "asking_price": price,
        })
    return rows


def main() -> None:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
