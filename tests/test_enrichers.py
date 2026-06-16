#!/usr/bin/env python3
"""Unit tests for enricher extraction logic — text in, structured fields out.

features_enricher.extract_amenities(text) is pure regex over description text,
so it's tested directly with representative strings (no fixture files needed).

These guard the amenity/floor/furnished extraction the model features depend on;
if the regexes drift, ML feature parity (test_js_python_parity) would silently
degrade — these catch it at the source.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from property_scraper.spiders.features_enricher import FeaturesEnricherSpider


@pytest.fixture(scope="module")
def enricher():
    # Constructor runs standalone (reads DB path but extract_amenities needs no DB).
    return FeaturesEnricherSpider()


class TestAmenityExtraction:
    def test_outdoor_space(self, enricher):
        a = enricher.extract_amenities(
            "A stunning flat with a private balcony and a roof terrace overlooking the garden."
        )
        assert a["has_balcony"] is True
        assert a["has_roof_terrace"] is True
        assert a["has_terrace"] is True  # roof terrace also sets terrace
        assert a["has_garden"] is True

    def test_building_features(self, enricher):
        a = enricher.extract_amenities(
            "Period conversion with a lift, 24-hour concierge, on-site gym and swimming pool."
        )
        assert a["has_lift"] is True
        assert a["has_porter"] is True
        assert a["has_gym"] is True
        assert a["has_pool"] is True

    def test_parking(self, enricher):
        assert enricher.extract_amenities("Allocated parking space included.")["has_parking"] is True
        assert enricher.extract_amenities("Private garage to the rear.")["has_parking"] is True
        assert enricher.extract_amenities("No car space available.")["has_parking"] is True  # 'car space'

    def test_floor_level_extraction(self, enricher):
        assert enricher.extract_amenities("Located on the 3rd floor.")["floor_level"] == 3
        assert enricher.extract_amenities("A ground floor apartment.")["floor_level"] == 0
        assert enricher.extract_amenities("Spacious basement flat.")["floor_level"] == -1
        assert enricher.extract_amenities("Exceptional penthouse.")["floor_level"] == 99

    def test_no_false_positive_amenities(self, enricher):
        a = enricher.extract_amenities("A plain one bedroom flat with no special features.")
        assert a["has_balcony"] is False
        assert a["has_gym"] is False
        assert a["has_pool"] is False
        assert a["has_lift"] is False
        assert a["floor_level"] is None

    def test_furnished_status(self, enricher):
        assert enricher.extract_amenities("Offered unfurnished.")["furnished"] == "unfurnished"
        assert enricher.extract_amenities("Beautifully furnished throughout.")["furnished"] == "furnished"
        assert enricher.extract_amenities("Plain text, no mention.")["furnished"] is None

    def test_climate_features(self, enricher):
        a = enricher.extract_amenities("Comfort cooling / air-conditioning and underfloor heating.")
        assert a["has_air_conditioning"] is True
        assert a["has_underfloor_heating"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
