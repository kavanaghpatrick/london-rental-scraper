#!/usr/bin/env python3
"""Tests for address fingerprinting service."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from property_scraper.services.fingerprint import (
    normalize_postcode,
    extract_street_number_and_unit,
    normalize_street_name,
    generate_fingerprint,
    parse_address,
    expand_saint,
)


class TestNormalizePostcode:
    def test_removes_spaces(self):
        assert normalize_postcode("SW1A 1AA") == "sw1a1aa"

    def test_already_normalized(self):
        assert normalize_postcode("sw1a1aa") == "sw1a1aa"

    def test_multiple_spaces(self):
        assert normalize_postcode("SW1A  1AA") == "sw1a1aa"

    def test_empty(self):
        assert normalize_postcode("") == ""
        assert normalize_postcode(None) == ""


class TestExpandSaint:
    def test_expands_st_before_name(self):
        assert "saint" in expand_saint("St John's Wood").lower()

    def test_does_not_expand_suffix_st(self):
        # This is tricky - "High St" should NOT expand
        # Our function only expands when followed by uppercase
        result = expand_saint("High St")
        assert "saint" not in result.lower()


class TestExtractStreetNumberAndUnit:
    def test_simple_number(self):
        num, unit = extract_street_number_and_unit("123 High Street")
        assert num == "123"
        assert unit == ""

    def test_with_letter_suffix(self):
        # UPDATED (Gemini review fix): Letter suffix is now part of street_num
        # to prevent hash collisions between 100a and 100b buildings
        num, unit = extract_street_number_and_unit("123a High Street")
        assert num == "123a"
        assert unit == ""  # Letter suffix not treated as unit

    def test_flat_prefix(self):
        num, unit = extract_street_number_and_unit("Flat 1, 123 High Street")
        assert num == "123"
        assert unit == "1"

    def test_apartment_prefix(self):
        num, unit = extract_street_number_and_unit("Apartment 5, 456 Victoria Road")
        assert num == "456"
        assert unit == "5"

    def test_basement_textual(self):
        num, unit = extract_street_number_and_unit("Basement, 10 High Street")
        assert num == "10"
        assert unit == "basement"

    def test_ground_floor(self):
        num, unit = extract_street_number_and_unit("Ground Floor Flat, 15 Victoria Road")
        assert num == "15"
        assert unit == "ground"

    def test_no_number(self):
        num, unit = extract_street_number_and_unit("Victoria Mansions")
        assert num == ""


class TestNormalizeStreetName:
    def test_simple_with_suffix(self):
        result = normalize_street_name("123 High Street")
        assert "high" in result
        assert "street" in result

    def test_preserves_different_suffixes(self):
        street = normalize_street_name("456 Victoria Road")
        road = normalize_street_name("789 Victoria Street")
        assert "road" in street
        assert "street" in road
        # They should be different!
        assert street != road

    def test_saint_expansion(self):
        result = normalize_street_name("St John's Wood Road")
        assert "saint" in result or "john" in result

    def test_with_flat_prefix(self):
        result = normalize_street_name("Flat 1, 123 High Street")
        assert "high" in result
        assert "street" in result


class TestGenerateFingerprint:
    def test_same_address_same_fingerprint(self):
        fp1 = generate_fingerprint("123 High Street", "SW1A 1AA")
        fp2 = generate_fingerprint("123 High Street", "sw1a1aa")
        assert fp1 == fp2

    def test_different_addresses_different_fingerprints(self):
        fp1 = generate_fingerprint("123 High Street", "SW1A 1AA")
        fp2 = generate_fingerprint("456 High Street", "SW1A 1AA")
        assert fp1 != fp2

    def test_different_postcodes_different_fingerprints(self):
        fp1 = generate_fingerprint("123 High Street", "SW1A 1AA")
        fp2 = generate_fingerprint("123 High Street", "W1A 1AA")
        assert fp1 != fp2

    def test_different_units_different_fingerprints(self):
        fp1 = generate_fingerprint("Flat 1, 123 High Street", "SW1A 1AA")
        fp2 = generate_fingerprint("Flat 2, 123 High Street", "SW1A 1AA")
        assert fp1 != fp2

    def test_length_is_16(self):
        fp = generate_fingerprint("123 High Street", "SW1A 1AA")
        assert len(fp) == 16


class TestCollisionPrevention:
    """Gemini-recommended tests for collision prevention."""

    def test_different_street_types_different_fingerprints(self):
        """10 High Street vs 10 High Road should NOT match."""
        fp_street = generate_fingerprint("10 High Street", "SW1A 1AA")
        fp_road = generate_fingerprint("10 High Road", "SW1A 1AA")
        assert fp_street != fp_road, "Street vs Road should produce different fingerprints"

    def test_different_suffix_variations(self):
        """Test various suffix pairs don't collide."""
        base = "Victoria"
        pc = "SW1A 1AA"
        fingerprints = set()

        for suffix in ["Street", "Road", "Lane", "Avenue", "Drive", "Place"]:
            fp = generate_fingerprint(f"10 {base} {suffix}", pc)
            assert fp not in fingerprints, f"{suffix} collided with another suffix"
            fingerprints.add(fp)


class TestTextualUnits:
    """Gemini-recommended tests for textual units."""

    def test_basement_flat(self):
        num, unit = extract_street_number_and_unit("Basement Flat, 10 High Street")
        assert unit == "basement"
        assert num == "10"

    def test_ground_floor_flat(self):
        num, unit = extract_street_number_and_unit("Ground Floor, 10 High Street")
        assert unit == "ground"
        assert num == "10"

    def test_penthouse(self):
        num, unit = extract_street_number_and_unit("Penthouse, 100 Park Lane")
        assert unit == "penthouse"
        assert num == "100"


class TestLetterSuffixSplit:
    """Test that 123a keeps suffix in street_num (prevents hash collisions)."""

    def test_letter_suffix_split(self):
        # UPDATED (Gemini review fix): Letter suffix stays with street_num
        # to prevent "Flat 1, 100a" and "Flat 1, 100b" from colliding
        num, unit = extract_street_number_and_unit("123A High Street")
        assert num == "123a"  # Suffix included in street_num
        assert unit == ""  # Not used as unit

    def test_100a_vs_100b_different_fingerprints(self):
        """100a High St and 100b High St should produce DIFFERENT fingerprints."""
        # This is the key test - the bug that was fixed
        fp1 = generate_fingerprint("Flat 1, 100a High Street", "SW1A 1AA")
        fp2 = generate_fingerprint("Flat 1, 100b High Street", "SW1A 1AA")
        # These are different buildings, must have different fingerprints!
        assert fp1 != fp2


class TestRealWorldCases:
    """Tests based on actual data patterns from the database."""

    def test_savills_format(self):
        fp = generate_fingerprint(
            "Flat 12, Eaton Place Mansions, 16 Eaton Place",
            "SW1X 8BY"
        )
        assert len(fp) == 16

    def test_knight_frank_format(self):
        fp = generate_fingerprint(
            "4 South Eaton Place, Belgravia, London SW1W 9JA",
            "SW1W 9JA"
        )
        assert len(fp) == 16

    def test_rightmove_abbreviated(self):
        fp = generate_fingerprint(
            "16 Eaton Pl, Flat 12",
            "SW1X8BY"
        )
        assert len(fp) == 16

    def test_empty_inputs(self):
        """Should not crash on empty/None inputs."""
        fp = generate_fingerprint("", "")
        assert fp  # Should still produce a hash
        fp2 = generate_fingerprint(None, None)
        assert fp2


class TestParseAddress:
    def test_parse_returns_dict(self):
        result = parse_address("Flat 1, 123 High Street", "SW1A 1AA")
        assert 'postcode' in result
        assert 'street_number' in result
        assert 'street_name' in result
        assert 'unit' in result
        assert 'fingerprint' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
