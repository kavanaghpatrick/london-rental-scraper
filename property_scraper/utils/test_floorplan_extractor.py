"""
Test suite for FloorplanExtractor based on visual audit ground truth.

Run with: python3 -m pytest property_scraper/utils/test_floorplan_extractor.py -v

Ground truth data collected from visual inspection of 20 John D Wood floorplans on 2025-12-06.
"""

import pytest
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

# Import the extractor
import sys
from pathlib import Path
# Add the utils directory to path for import
sys.path.insert(0, str(Path(__file__).parent))
from floorplan_extractor import FloorplanExtractor, FloorplanData, FloorData


# Ground truth from visual audit
@dataclass
class GroundTruth:
    property_id: str
    address: str
    expected_sqft: Optional[int]
    expected_floor_count: int
    expected_property_levels: str  # single_floor, duplex, triplex, multi_floor
    expected_floors: List[str]  # List of floor names visible in image
    expected_bedrooms: int
    notes: str = ""


# All 20 properties with ground truth from visual audit
GROUND_TRUTH = [
    GroundTruth(
        property_id="20324595",
        address="57A Princes Gate SW7",
        expected_sqft=7187,
        expected_floor_count=4,
        expected_property_levels="multi_floor",
        expected_floors=["Basement", "Lower Ground Floor", "Ground Floor", "Mezzanine"],
        expected_bedrooms=5,
        notes="4 levels including basement"
    ),
    GroundTruth(
        property_id="20620728",
        address="Glebe Place Chelsea",
        expected_sqft=4000,
        expected_floor_count=6,
        expected_property_levels="multi_floor",
        expected_floors=["Basement", "Lower Ground Floor", "Ground Floor", "First Floor", "Second Floor", "Third Floor"],
        expected_bedrooms=6,
        notes="6-level townhouse, DB incorrectly had 0 floors"
    ),
    GroundTruth(
        property_id="20639639",
        address="57A Princes Gate SW7",
        expected_sqft=7187,
        expected_floor_count=4,
        expected_property_levels="multi_floor",
        expected_floors=["Basement", "Lower Ground Floor", "Ground Floor", "Mezzanine"],
        expected_bedrooms=5,
        notes="Duplicate of 20324595"
    ),
    GroundTruth(
        property_id="20678425",
        address="Quernsmuir Cottage, Sands Road",
        expected_sqft=4090,
        expected_floor_count=3,
        expected_property_levels="triplex",
        expected_floors=["Lower Ground Floor", "Ground Floor", "First Floor"],
        expected_bedrooms=5,
        notes="Plus Pool House and Garden Room outbuildings"
    ),
    GroundTruth(
        property_id="20724108",
        address="Queens Drive Oxshott",
        expected_sqft=10443,
        expected_floor_count=3,
        expected_property_levels="triplex",
        expected_floors=["Ground Floor", "First Floor", "Lower Floor"],
        expected_bedrooms=8,
        notes="Lower floor has pool and cinema"
    ),
    GroundTruth(
        property_id="20773274",
        address="Forestry Road",
        expected_sqft=5582,  # Main house only, outbuildings add 4752
        expected_floor_count=3,
        expected_property_levels="triplex",
        expected_floors=["Ground Floor", "First Floor", "Second Floor"],
        expected_bedrooms=5,
        notes="Plus outbuildings"
    ),
    GroundTruth(
        property_id="20877906",
        address="Sussex Street Pimlico",
        expected_sqft=3251,
        expected_floor_count=5,
        expected_property_levels="multi_floor",
        expected_floors=["Lower Ground Floor", "Ground Floor", "First Floor", "Second Floor", "Third Floor"],
        expected_bedrooms=5,
        notes="5 floors + roof terrace"
    ),
    GroundTruth(
        property_id="20899588",
        address="Eaton Square SW1W",
        expected_sqft=1231,
        expected_floor_count=1,
        expected_property_levels="single_floor",
        expected_floors=["Raised Ground Floor"],
        expected_bedrooms=2,
        notes="Single floor apartment - all data correct"
    ),
    GroundTruth(
        property_id="20917096",
        address="Burnsall Street Chelsea",
        expected_sqft=2314,
        expected_floor_count=4,
        expected_property_levels="multi_floor",
        expected_floors=["Lower Ground Floor", "Ground Floor", "First Floor", "Second Floor"],
        expected_bedrooms=4,
        notes="DB incorrectly had 3 floors"
    ),
    GroundTruth(
        property_id="20940875",
        address="South Eaton Place SW1W",
        expected_sqft=4875,
        expected_floor_count=7,
        expected_property_levels="multi_floor",
        expected_floors=["Lower Ground Floor", "Ground Floor", "First Floor", "Second Floor", "Third Floor", "Fourth Floor", "Fifth Floor"],
        expected_bedrooms=7,
        notes="7-floor townhouse"
    ),
    GroundTruth(
        property_id="20940883",
        address="South Eaton Place SW1W",
        expected_sqft=4875,
        expected_floor_count=7,
        expected_property_levels="multi_floor",
        expected_floors=["Lower Ground Floor", "Ground Floor", "First Floor", "Second Floor", "Third Floor", "Fourth Floor", "Fifth Floor"],
        expected_bedrooms=7,
        notes="Duplicate of 20940875"
    ),
    GroundTruth(
        property_id="20964078",
        address="Chelsea Square",
        expected_sqft=5004,
        expected_floor_count=4,
        expected_property_levels="multi_floor",
        expected_floors=["Ground Floor", "First Floor", "Second Floor", "Third Floor"],
        expected_bedrooms=3,
        notes="DB incorrectly had 2 floors and duplex"
    ),
    GroundTruth(
        property_id="20980534",
        address="Sloane Gardens Chelsea",
        expected_sqft=1540,  # NOT 4540 as in DB - critical error
        expected_floor_count=2,
        expected_property_levels="duplex",
        expected_floors=["Lower Ground Floor", "Raised Ground Floor"],
        expected_bedrooms=3,
        notes="CRITICAL: DB had 4540, actual is 1540"
    ),
    GroundTruth(
        property_id="21061073",
        address="Priory Lane SW15",
        expected_sqft=5082,
        expected_floor_count=3,
        expected_property_levels="triplex",
        expected_floors=["Ground Floor", "First Floor", "Second Floor"],
        expected_bedrooms=6,
        notes="DB incorrectly had 7 bedrooms"
    ),
    GroundTruth(
        property_id="21061074",
        address="Priory Lane SW15",
        expected_sqft=5082,
        expected_floor_count=3,
        expected_property_levels="triplex",
        expected_floors=["Ground Floor", "First Floor", "Second Floor"],
        expected_bedrooms=6,
        notes="Duplicate of 21061073"
    ),
    GroundTruth(
        property_id="21071779",
        address="St Mary Abbots Terrace W14",
        expected_sqft=1855,
        expected_floor_count=3,
        expected_property_levels="triplex",
        expected_floors=["Ground Floor", "First Floor", "Second Floor"],
        expected_bedrooms=4,
        notes="DB incorrectly had 2 floors and duplex"
    ),
    GroundTruth(
        property_id="21090586",
        address="Cadogan Square SW1X",
        expected_sqft=1987,
        expected_floor_count=3,
        expected_property_levels="triplex",
        expected_floors=["Fourth Floor", "Fifth Floor", "Sixth Floor"],
        expected_bedrooms=3,
        notes="All data correct"
    ),
    GroundTruth(
        property_id="21101528",
        address="Hans Road Knightsbridge",
        expected_sqft=3696,
        expected_floor_count=1,
        expected_property_levels="single_floor",
        expected_floors=["Fourth Floor"],
        expected_bedrooms=5,
        notes="DB had NULL sqft"
    ),
    GroundTruth(
        property_id="21105050",
        address="The View Palace Street",
        expected_sqft=4627,
        expected_floor_count=3,
        expected_property_levels="triplex",
        expected_floors=["Fifteenth Floor", "Sixteenth Floor", "Seventeenth Floor"],
        expected_bedrooms=5,
        notes="DB had NULL sqft and 0 floors"
    ),
    GroundTruth(
        property_id="21235558",
        address="Hereford Square SW7",
        expected_sqft=2167,
        expected_floor_count=3,
        expected_property_levels="triplex",
        expected_floors=["Ground Floor", "First Floor", "Second Floor"],
        expected_bedrooms=3,
        notes="DB had unknown floor_count"
    ),
]

# Path to test images
TEST_IMAGE_DIR = Path("/tmp/floorplan_audit")


@pytest.fixture
def extractor():
    """Create FloorplanExtractor instance."""
    return FloorplanExtractor()


def get_image_path(property_id: str) -> Optional[Path]:
    """Get image path for a property, trying both .jpg and .png extensions."""
    for ext in ['.jpg', '.png']:
        path = TEST_IMAGE_DIR / f"{property_id}{ext}"
        if path.exists():
            return path
    return None


class TestSqftExtraction:
    """Test square footage extraction accuracy."""

    @pytest.mark.parametrize("gt", GROUND_TRUTH)
    def test_sqft_extraction(self, extractor, gt: GroundTruth):
        """Test sqft is extracted within 5% tolerance."""
        image_path = get_image_path(gt.property_id)
        if not image_path:
            pytest.skip(f"Image not found for {gt.property_id}")

        if gt.expected_sqft is None:
            pytest.skip(f"No expected sqft for {gt.property_id}")

        result = extractor.extract_from_file(str(image_path))

        # Allow 5% tolerance for OCR errors
        tolerance = 0.05
        lower_bound = gt.expected_sqft * (1 - tolerance)
        upper_bound = gt.expected_sqft * (1 + tolerance)

        assert result.total_sqft is not None, \
            f"Failed to extract sqft from {gt.property_id} ({gt.address})"

        assert lower_bound <= result.total_sqft <= upper_bound, \
            f"Property {gt.property_id}: Expected {gt.expected_sqft} sqft, got {result.total_sqft} " \
            f"(tolerance: {lower_bound:.0f}-{upper_bound:.0f})"


class TestFloorCounting:
    """Test floor count extraction accuracy."""

    @pytest.mark.parametrize("gt", GROUND_TRUTH)
    def test_floor_count(self, extractor, gt: GroundTruth):
        """Test floor count matches ground truth."""
        image_path = get_image_path(gt.property_id)
        if not image_path:
            pytest.skip(f"Image not found for {gt.property_id}")

        result = extractor.extract_from_file(str(image_path))

        assert result.floor_data is not None, \
            f"Failed to extract floor data from {gt.property_id}"

        assert result.floor_data.floor_count == gt.expected_floor_count, \
            f"Property {gt.property_id} ({gt.address}): " \
            f"Expected {gt.expected_floor_count} floors, got {result.floor_data.floor_count}. " \
            f"Expected floors: {gt.expected_floors}. " \
            f"Detected floors: {result.floor_data.floors_raw}"

    @pytest.mark.parametrize("gt", GROUND_TRUTH)
    def test_property_levels_classification(self, extractor, gt: GroundTruth):
        """Test property_levels classification matches ground truth."""
        image_path = get_image_path(gt.property_id)
        if not image_path:
            pytest.skip(f"Image not found for {gt.property_id}")

        result = extractor.extract_from_file(str(image_path))

        assert result.floor_data is not None, \
            f"Failed to extract floor data from {gt.property_id}"

        assert result.floor_data.property_levels == gt.expected_property_levels, \
            f"Property {gt.property_id} ({gt.address}): " \
            f"Expected '{gt.expected_property_levels}', got '{result.floor_data.property_levels}'. " \
            f"Floor count: {result.floor_data.floor_count}"


class TestSpecificFloorDetection:
    """Test detection of specific floor types."""

    def test_basement_detection(self, extractor):
        """Test basement floors are detected."""
        # Properties with basements: 20324595, 20620728, 20639639
        for pid in ["20324595", "20620728", "20639639"]:
            image_path = get_image_path(pid)
            if not image_path:
                continue

            result = extractor.extract_from_file(str(image_path))
            assert result.floor_data.has_basement == 1 or result.floor_data.has_lower_ground == 1, \
                f"Property {pid} should have basement or lower ground detected"

    def test_lower_ground_detection(self, extractor):
        """Test lower ground floors are detected."""
        # Properties with lower ground: 20917096, 20980534, 20877906
        for pid in ["20917096", "20980534", "20877906"]:
            image_path = get_image_path(pid)
            if not image_path:
                continue

            result = extractor.extract_from_file(str(image_path))
            assert result.floor_data.has_lower_ground == 1, \
                f"Property {pid} should have lower_ground detected. " \
                f"Floors found: {result.floor_data.floors_raw}"

    def test_high_floor_numbers(self, extractor):
        """Test high floor numbers (4th+) are detected."""
        # Property 21090586 has 4th, 5th, 6th floors
        image_path = get_image_path("21090586")
        if not image_path:
            pytest.skip("Image not found for 21090586")

        result = extractor.extract_from_file(str(image_path))
        assert result.floor_data.has_fourth_plus == 1, \
            f"Property 21090586 should have 4th+ floors detected. " \
            f"Floors found: {result.floor_data.floors_raw}"

    def test_very_high_floors(self, extractor):
        """Test very high floor numbers (15th+) are detected."""
        # Property 21105050 has 15th, 16th, 17th floors
        image_path = get_image_path("21105050")
        if not image_path:
            pytest.skip("Image not found for 21105050")

        result = extractor.extract_from_file(str(image_path))
        # Should detect these as high floors
        assert result.floor_data.floor_count >= 3, \
            f"Property 21105050 should have at least 3 floors. " \
            f"Floors found: {result.floor_data.floors_raw}"


class TestEdgeCases:
    """Test edge cases and known problem areas."""

    def test_sloane_gardens_sqft_not_misread(self, extractor):
        """Critical: Ensure 1540 is not misread as 4540."""
        image_path = get_image_path("20980534")
        if not image_path:
            pytest.skip("Image not found for 20980534")

        result = extractor.extract_from_file(str(image_path))

        # This was the critical error: 1540 misread as 4540
        assert result.total_sqft is not None
        assert result.total_sqft < 3000, \
            f"Sloane Gardens sqft should be ~1540, not {result.total_sqft}. " \
            f"This is a known OCR misread issue."

    def test_six_floor_townhouse(self, extractor):
        """Test 6-floor townhouse is detected correctly."""
        # Property 20620728 has 6 distinct floors
        image_path = get_image_path("20620728")
        if not image_path:
            pytest.skip("Image not found for 20620728")

        result = extractor.extract_from_file(str(image_path))

        assert result.floor_data.floor_count >= 5, \
            f"Glebe Place should have 5-6 floors detected, got {result.floor_data.floor_count}. " \
            f"Floors: {result.floor_data.floors_raw}"

        assert result.floor_data.property_levels == "multi_floor", \
            f"6-floor property should be 'multi_floor', got '{result.floor_data.property_levels}'"

    def test_mezzanine_counted_as_floor(self, extractor):
        """Test mezzanine levels are counted in floor total."""
        # Property 20324595 has mezzanine
        image_path = get_image_path("20324595")
        if not image_path:
            pytest.skip("Image not found for 20324595")

        result = extractor.extract_from_file(str(image_path))

        # Should count mezzanine as a floor
        assert result.floor_data.has_mezzanine == 1, \
            f"Princes Gate should have mezzanine detected"


class TestRawTextExtraction:
    """Test that OCR raw text contains expected content."""

    @pytest.mark.parametrize("gt", GROUND_TRUTH[:5])  # Test first 5 for speed
    def test_raw_text_not_empty(self, extractor, gt: GroundTruth):
        """Test that raw OCR text is extracted."""
        image_path = get_image_path(gt.property_id)
        if not image_path:
            pytest.skip(f"Image not found for {gt.property_id}")

        result = extractor.extract_from_file(str(image_path))

        assert len(result.raw_text) > 50, \
            f"Raw text too short for {gt.property_id}: {len(result.raw_text)} chars"

    def test_floor_labels_in_raw_text(self, extractor):
        """Test that floor labels appear in raw text."""
        image_path = get_image_path("20877906")  # Sussex Street - 5 floors
        if not image_path:
            pytest.skip("Image not found")

        result = extractor.extract_from_file(str(image_path))
        text_lower = result.raw_text.lower()

        # Should find at least some floor references
        floor_keywords = ["ground", "first", "second", "floor"]
        found = sum(1 for kw in floor_keywords if kw in text_lower)
        assert found >= 2, \
            f"Expected floor labels in raw text. Found keywords: {found}"


class TestAccuracyMetrics:
    """Calculate overall accuracy metrics."""

    def test_sqft_accuracy_above_threshold(self, extractor):
        """Test overall sqft extraction accuracy is above 75%."""
        correct = 0
        total = 0
        errors = []

        for gt in GROUND_TRUTH:
            image_path = get_image_path(gt.property_id)
            if not image_path or gt.expected_sqft is None:
                continue

            total += 1
            result = extractor.extract_from_file(str(image_path))

            if result.total_sqft is not None:
                tolerance = 0.05
                lower = gt.expected_sqft * (1 - tolerance)
                upper = gt.expected_sqft * (1 + tolerance)
                if lower <= result.total_sqft <= upper:
                    correct += 1
                else:
                    errors.append(f"{gt.property_id}: expected {gt.expected_sqft}, got {result.total_sqft}")
            else:
                errors.append(f"{gt.property_id}: NULL (expected {gt.expected_sqft})")

        accuracy = correct / total if total > 0 else 0
        print(f"\nSqft accuracy: {correct}/{total} = {accuracy:.1%}")
        for e in errors:
            print(f"  Error: {e}")

        assert accuracy >= 0.75, \
            f"Sqft accuracy {accuracy:.1%} below 75% threshold"

    def test_floor_count_accuracy_above_threshold(self, extractor):
        """Test overall floor count accuracy is above 70%."""
        correct = 0
        total = 0
        errors = []

        for gt in GROUND_TRUTH:
            image_path = get_image_path(gt.property_id)
            if not image_path:
                continue

            total += 1
            result = extractor.extract_from_file(str(image_path))

            if result.floor_data and result.floor_data.floor_count == gt.expected_floor_count:
                correct += 1
            else:
                actual = result.floor_data.floor_count if result.floor_data else "None"
                errors.append(f"{gt.property_id}: expected {gt.expected_floor_count}, got {actual}")

        accuracy = correct / total if total > 0 else 0
        print(f"\nFloor count accuracy: {correct}/{total} = {accuracy:.1%}")
        for e in errors:
            print(f"  Error: {e}")

        assert accuracy >= 0.70, \
            f"Floor count accuracy {accuracy:.1%} below 70% threshold"


if __name__ == "__main__":
    # Quick standalone test
    import sys

    extractor = FloorplanExtractor()
    print("=" * 60)
    print("FLOORPLAN EXTRACTOR TEST SUMMARY")
    print("=" * 60)

    sqft_correct = 0
    floor_correct = 0
    total = 0

    for gt in GROUND_TRUTH:
        image_path = get_image_path(gt.property_id)
        if not image_path:
            print(f"SKIP {gt.property_id}: Image not found")
            continue

        total += 1
        result = extractor.extract_from_file(str(image_path))

        sqft_ok = "FAIL"
        if result.total_sqft and gt.expected_sqft:
            tol = 0.05
            if gt.expected_sqft * (1-tol) <= result.total_sqft <= gt.expected_sqft * (1+tol):
                sqft_ok = "PASS"
                sqft_correct += 1

        floor_ok = "FAIL"
        if result.floor_data and result.floor_data.floor_count == gt.expected_floor_count:
            floor_ok = "PASS"
            floor_correct += 1

        print(f"{gt.property_id}: sqft={sqft_ok} ({result.total_sqft} vs {gt.expected_sqft}) | "
              f"floors={floor_ok} ({result.floor_data.floor_count if result.floor_data else 'N/A'} vs {gt.expected_floor_count})")

    print("=" * 60)
    print(f"SQFT:   {sqft_correct}/{total} = {sqft_correct/total*100:.1f}%")
    print(f"FLOORS: {floor_correct}/{total} = {floor_correct/total*100:.1f}%")
    print("=" * 60)

    sys.exit(0 if sqft_correct/total >= 0.75 and floor_correct/total >= 0.70 else 1)
