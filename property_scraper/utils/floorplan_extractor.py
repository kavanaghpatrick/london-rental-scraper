"""
Floorplan Extractor - Extract structured data from floorplan images using OCR

This module provides comprehensive extraction of property data from floorplan images:
- Total area (sqft, sqm)
- Individual room dimensions
- Room counts by type
- Floor levels
- Ceiling heights
- Outdoor spaces

- Special features

Usage:
    from property_scraper.utils.floorplan_extractor import FloorplanExtractor

    extractor = FloorplanExtractor()

    # From file path
    data = extractor.extract_from_file('/path/to/floorplan.jpg')

    # From PIL Image
    data = extractor.extract_from_image(pil_image)

    # From bytes
    data = extractor.extract_from_bytes(image_bytes)

    # From URL (async)
    data = await extractor.extract_from_url('https://...')

Returns:
    {
        'total_sqft': 2167,
        'total_sqm': 201,
        'rooms': [
            {'type': 'bedroom', 'name': 'Principal Bedroom', 'dimensions': "19'9 x 13'7", 'sqft': 268},
            {'type': 'kitchen', 'name': 'Kitchen', 'dimensions': "16'11 x 11'0", 'sqft': 186},
            ...
        ],
        'room_counts': {'bedroom': 3, 'bathroom': 2, 'reception': 1},
        'floors': ['Ground Floor', 'First Floor', 'Second Floor'],
        'outdoor_spaces': [
            {'type': 'terrace', 'dimensions': "20'8 x 11'5", 'sqft': 236}
        ],
        'special_features': ['garage', 'study'],
        'extraction_confidence': 0.95,
        'raw_text': '...'
    }

Requirements:
    pip install pytesseract Pillow
    brew install tesseract  # macOS
"""
from __future__ import annotations  # Defers annotation evaluation (fixes PIL type hints)

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from io import BytesIO

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    Image = None  # Type stub for when PIL not available


@dataclass
class Room:
    """Represents a room extracted from floorplan."""
    type: str  # bedroom, bathroom, kitchen, reception, etc.
    name: str  # Full name like "Principal Bedroom"
    dimensions_imperial: Optional[str] = None  # "19'9 x 13'7"
    dimensions_metric: Optional[str] = None  # "6.02 x 4.14"
    sqft: Optional[int] = None
    sqm: Optional[float] = None
    floor: Optional[str] = None  # "Ground Floor"
    ceiling_height: Optional[str] = None  # "2.54m" or "8'6"
    features: List[str] = field(default_factory=list)  # ['en-suite', 'built-in storage']


@dataclass
class OutdoorSpace:
    """Represents an outdoor space extracted from floorplan."""
    type: str  # terrace, garden, balcony, patio
    dimensions_imperial: Optional[str] = None
    dimensions_metric: Optional[str] = None
    sqft: Optional[int] = None
    sqm: Optional[float] = None


@dataclass
class FloorData:
    """Binary floor classification for ML model training."""
    has_basement: int = 0
    has_lower_ground: int = 0
    has_ground: int = 0
    has_mezzanine: int = 0
    has_first_floor: int = 0
    has_second_floor: int = 0
    has_third_floor: int = 0
    has_fourth_plus: int = 0
    has_roof_terrace: int = 0
    floor_count: int = 0
    property_levels: str = "unknown"  # single_floor, duplex, triplex, multi_floor
    floors_raw: List[str] = field(default_factory=list)  # Original floor names


@dataclass
class FloorplanData:
    """Complete extracted data from a floorplan."""
    total_sqft: Optional[int] = None
    total_sqm: Optional[int] = None
    rooms: List[Room] = field(default_factory=list)
    room_counts: Dict[str, int] = field(default_factory=dict)
    floors: List[str] = field(default_factory=list)
    floor_data: Optional[FloorData] = None  # Binary floor classification
    outdoor_spaces: List[OutdoorSpace] = field(default_factory=list)
    special_features: List[str] = field(default_factory=list)
    address: Optional[str] = None
    extraction_confidence: float = 0.0
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert Room and OutdoorSpace objects to dicts
        data['rooms'] = [asdict(r) for r in self.rooms]
        data['outdoor_spaces'] = [asdict(o) for o in self.outdoor_spaces]
        if self.floor_data:
            data['floor_data'] = asdict(self.floor_data)
        return data


class FloorplanExtractor:
    """Extract structured data from floorplan images using OCR."""

    # Floor standardization mapping: pattern -> (canonical_name, display_name)
    FLOOR_PATTERNS = {
        # Basement variations
        r'basement': ('basement', 'Basement'),
        r'cellar': ('basement', 'Basement'),
        # Lower ground variations
        r'lower\s*ground\s*(?:floor)?': ('lower_ground', 'Lower Ground'),
        r'lgf': ('lower_ground', 'Lower Ground'),
        r'lower\s*level': ('lower_ground', 'Lower Ground'),
        # Ground floor variations
        r'ground\s*(?:floor)?': ('ground', 'Ground Floor'),
        r'gf\b': ('ground', 'Ground Floor'),
        r'street\s*level': ('ground', 'Ground Floor'),
        # Mezzanine
        r'mezzanine': ('mezzanine', 'Mezzanine'),
        r'mezz\b': ('mezzanine', 'Mezzanine'),
        # First floor variations
        r'first\s*(?:floor)?': ('first', 'First Floor'),
        r'1st\s*(?:floor)?': ('first', 'First Floor'),
        # Second floor variations
        r'second\s*(?:floor)?': ('second', 'Second Floor'),
        r'2nd\s*(?:floor)?': ('second', 'Second Floor'),
        # Third floor variations
        r'third\s*(?:floor)?': ('third', 'Third Floor'),
        r'3rd\s*(?:floor)?': ('third', 'Third Floor'),
        # Fourth floor variations
        r'fourth\s*(?:floor)?': ('fourth', 'Fourth Floor'),
        r'4th\s*(?:floor)?': ('fourth', 'Fourth Floor'),
        # Fifth floor variations
        r'fifth\s*(?:floor)?': ('fifth', 'Fifth Floor'),
        r'5th\s*(?:floor)?': ('fifth', 'Fifth Floor'),
        # Sixth floor and above
        r'sixth\s*(?:floor)?': ('sixth', 'Sixth Floor'),
        r'6th\s*(?:floor)?': ('sixth', 'Sixth Floor'),
        r'seventh\s*(?:floor)?': ('seventh', 'Seventh Floor'),
        r'7th\s*(?:floor)?': ('seventh', 'Seventh Floor'),
        # Eighth through twentieth
        r'eighth\s*(?:floor)?': ('eighth', 'Eighth Floor'),
        r'8th\s*(?:floor)?': ('eighth', 'Eighth Floor'),
        r'ninth\s*(?:floor)?': ('ninth', 'Ninth Floor'),
        r'9th\s*(?:floor)?': ('ninth', 'Ninth Floor'),
        r'tenth\s*(?:floor)?': ('tenth', 'Tenth Floor'),
        r'10th\s*(?:floor)?': ('tenth', 'Tenth Floor'),
        r'eleventh\s*(?:floor)?': ('eleventh', 'Eleventh Floor'),
        r'11th\s*(?:floor)?': ('eleventh', 'Eleventh Floor'),
        r'twelfth\s*(?:floor)?': ('twelfth', 'Twelfth Floor'),
        r'12th\s*(?:floor)?': ('twelfth', 'Twelfth Floor'),
        r'thirteenth\s*(?:floor)?': ('thirteenth', 'Thirteenth Floor'),
        r'13th\s*(?:floor)?': ('thirteenth', 'Thirteenth Floor'),
        r'fourteenth\s*(?:floor)?': ('fourteenth', 'Fourteenth Floor'),
        r'14th\s*(?:floor)?': ('fourteenth', 'Fourteenth Floor'),
        r'fifteenth\s*(?:floor)?': ('fifteenth', 'Fifteenth Floor'),
        r'15th\s*(?:floor)?': ('fifteenth', 'Fifteenth Floor'),
        r'sixteenth\s*(?:floor)?': ('sixteenth', 'Sixteenth Floor'),
        r'16th\s*(?:floor)?': ('sixteenth', 'Sixteenth Floor'),
        r'seventeenth\s*(?:floor)?': ('seventeenth', 'Seventeenth Floor'),
        r'17th\s*(?:floor)?': ('seventeenth', 'Seventeenth Floor'),
        r'eighteenth\s*(?:floor)?': ('eighteenth', 'Eighteenth Floor'),
        r'18th\s*(?:floor)?': ('eighteenth', 'Eighteenth Floor'),
        r'nineteenth\s*(?:floor)?': ('nineteenth', 'Nineteenth Floor'),
        r'19th\s*(?:floor)?': ('nineteenth', 'Nineteenth Floor'),
        r'twentieth\s*(?:floor)?': ('twentieth', 'Twentieth Floor'),
        r'20th\s*(?:floor)?': ('twentieth', 'Twentieth Floor'),
        # Raised ground floor variation
        r'raised\s*ground\s*(?:floor)?': ('ground', 'Ground Floor'),
        # Roof/terrace variations
        r'roof\s*terrace': ('roof_terrace', 'Roof Terrace'),
        r'rooftop': ('roof_terrace', 'Roof Terrace'),
        r'roof\b': ('roof_terrace', 'Roof'),
        # Penthouse
        r'penthouse': ('penthouse', 'Penthouse'),
    }

    # Room type patterns
    ROOM_TYPES = {
        'bedroom': r'(?:master\s+)?(?:principal\s+)?bedroom\s*\d*|bed\s*\d+',
        'bathroom': r'bath(?:room)?\s*\d*|en[\-\s]?suite|shower\s*room|wc|cloakroom',
        'kitchen': r'kitchen|breakfast\s+room|kitchen/breakfast',
        'reception': r'reception(?:\s+room)?|living\s*room|sitting\s*room|lounge|drawing\s*room',
        'dining': r'dining\s*(?:room)?|dining/reception',
        'study': r'study|office|home\s+office',
        'utility': r'utility(?:\s+room)?|laundry(?:\s+room)?',
        'hallway': r'hall(?:way)?|entrance\s+hall|landing',
        'storage': r'storage|cupboard|wardrobe|dressing\s*(?:room|area)?',
        'garage': r'garage',
        'cellar': r'cellar|wine\s+cellar|basement',
        'cinema': r'cinema(?:\s+room)?|tv\s+room|media\s+room',
        'gym': r'gym|fitness|exercise',
        'pool': r'pool|swimming|spa',
        'plant': r'plant\s+room',
        'staff': r'staff\s+(?:bedroom|quarters|room)',
    }

    # Outdoor space patterns
    OUTDOOR_TYPES = {
        'terrace': r'terrace|roof\s+terrace',
        'garden': r'garden|rear\s+garden|front\s+garden',
        'balcony': r'balcony',
        'patio': r'patio|courtyard',
    }

    # Special features to look for
    SPECIAL_FEATURES = [
        'swimming pool', 'pool', 'spa', 'sauna', 'steam room', 'gym',
        'cinema', 'media room', 'wine cellar', 'cellar',
        'lift', 'elevator', 'garage', 'parking',
        'air conditioning', 'a/c', 'underfloor heating',
        'fireplace', 'wood burning',
    ]

    def __init__(self):
        if not OCR_AVAILABLE:
            raise RuntimeError(
                "pytesseract and Pillow required. Install with: pip install pytesseract Pillow"
            )

    def extract_from_file(self, file_path: str) -> FloorplanData:
        """Extract data from a floorplan image file."""
        img = Image.open(file_path)
        return self.extract_from_image(img)

    def extract_from_bytes(self, image_bytes: bytes) -> FloorplanData:
        """Extract data from image bytes."""
        img = Image.open(BytesIO(image_bytes))
        return self.extract_from_image(img)

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to grayscale
        gray = img.convert('L')

        # Increase contrast using autocontrast
        try:
            from PIL import ImageOps
            gray = ImageOps.autocontrast(gray, cutoff=1)
        except:
            pass

        # Scale up small images for better OCR
        width, height = gray.size
        if width < 1500 or height < 1500:
            scale = max(1500 / width, 1500 / height, 1.5)
            new_size = (int(width * scale), int(height * scale))
            gray = gray.resize(new_size, Image.Resampling.LANCZOS)

        return gray

    def _normalize_text(self, text: str) -> str:
        """Normalize common OCR typos and clean text."""
        # Common OCR typos map
        replacements = [
            (r'\bflocr\b', 'floor'),
            (r'\bfioor\b', 'floor'),
            (r'\bFloor\s*Flan\b', 'Floor Plan'),
            (r'\b2nth\b', '2nd'),
            (r'\b17th\s*Flocr\b', '17th Floor'),
            (r'\bseventeenth\s*flocr\b', 'seventeenth floor'),
        ]

        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.I)

        return result

    def _filter_exclusion_lines(self, text: str) -> str:
        """
        Remove lines containing exclusion context to avoid false floor detection.
        Returns text with exclusion lines removed.
        """
        exclusion_pattern = r'(?:excluding|exclude|not\s+included|reduced\s+headroom)'
        lines = text.split('\n')
        filtered_lines = []

        for line in lines:
            if not re.search(exclusion_pattern, line, re.I):
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def extract_from_image(self, img: Image.Image) -> FloorplanData:
        """Extract data from a PIL Image."""
        data = FloorplanData()

        # Preprocess image for better OCR
        processed_img = self._preprocess_image(img)

        # Get full OCR text - try processed first, then original
        full_text = pytesseract.image_to_string(processed_img)

        # If processed image yields little text, try original
        if len(full_text.strip()) < 50:
            original_text = pytesseract.image_to_string(img)
            if len(original_text.strip()) > len(full_text.strip()):
                full_text = original_text

        data.raw_text = full_text

        # Extract total area from header regions
        sqft, sqm = self._extract_total_area(img)
        data.total_sqft = sqft
        data.total_sqm = sqm

        # Extract address
        data.address = self._extract_address(full_text)

        # Extract rooms with dimensions
        data.rooms = self._extract_rooms(full_text)

        # Count rooms by type
        data.room_counts = self._count_rooms(data.rooms)

        # Extract floor levels
        data.floors = self._extract_floors(full_text)

        # Binary floor classification for ML
        data.floor_data = self._classify_floors(full_text)

        # Extract outdoor spaces
        data.outdoor_spaces = self._extract_outdoor_spaces(full_text)

        # Extract special features
        data.special_features = self._extract_special_features(full_text)

        # Calculate confidence score
        data.extraction_confidence = self._calculate_confidence(data)

        return data

    def _extract_total_area(self, img: Image.Image) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract total sqft/sqm using hierarchical approach:
        1. Explicit 'Total Gross Internal' / 'Total Area' patterns (highest priority)
        2. Any 'Total ... sq ft' pattern
        3. Largest sqft value found (heuristic fallback)
        """
        # Use preprocessed image for better OCR
        processed_img = self._preprocess_image(img)
        width, height = processed_img.size

        # Get full text once
        full_text = pytesseract.image_to_string(processed_img)

        # Also try original image if processed yields little text
        if len(full_text.strip()) < 50:
            original_text = pytesseract.image_to_string(img)
            if len(original_text.strip()) > len(full_text.strip()):
                full_text = original_text

        # Patterns
        sqft_pattern = r'(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|Sq\s*Ft|square\s*feet)'
        sqm_pattern = r'(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*m|sqm|Sq\s*M|square\s*m)'

        # Step 1: STRONGEST PATTERN - Look for explicit "Total = X sq ft" line first
        # This handles floorplans that show: "Main Area = 2038 sq ft" then "Garage = 129 sq ft" then "Total = 2167 sq ft"
        total_line_pattern = r'(?:^|\n)\s*Total\s*=?\s*[\d,\.]+\s*(?:sq\.?\s*m|sqm)\s*/?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)'
        total_match = re.search(total_line_pattern, full_text, re.I | re.M)
        if total_match:
            try:
                sqft = int(float(total_match.group(1).replace(',', '')))
                if 100 < sqft < 100000:
                    sqm = self._extract_sqm(full_text)
                    return sqft, sqm
            except (ValueError, IndexError):
                pass

        # Step 1b: More total/gross internal patterns
        # "Approximate Gross Internal Area = 4627 sq ft", "Total Area 1234 sq ft"
        strong_patterns = [
            r'(?:total|gross\s*internal|approximate)\s*(?:gross\s*internal\s*)?(?:area)?\s*(?:=|:|-|of)?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)',
            r'(?:total|overall)\s*(?:area)?\s*(?:=|:)?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)',
            r'(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)\s*(?:total|overall|gross)',
        ]

        for pattern in strong_patterns:
            matches = list(re.finditer(pattern, full_text, re.I))
            if matches:
                # If multiple matches, take the largest (likely the total vs individual floors)
                values = []
                for m in matches:
                    try:
                        val_str = m.group(1).replace(',', '')
                        val = float(val_str)
                        if 100 < val < 100000:
                            values.append(int(val))
                    except (ValueError, IndexError):
                        continue

                if values:
                    sqft = max(values)
                    sqm = self._extract_sqm(full_text)
                    return sqft, sqm

        # Step 2: Try header/footer regions where totals often appear
        regions = [
            (0, 0, width, int(height * 0.15)),  # Top 15%
            (0, int(height * 0.85), width, height),  # Bottom 15%
            (0, 0, int(width * 0.35), int(height * 0.25)),  # Top-left
            (int(width * 0.5), 0, width, int(height * 0.25)),  # Top-right
        ]

        for box in regions:
            try:
                crop = processed_img.crop(box)
                text = pytesseract.image_to_string(crop)

                # Look for total pattern in regions
                total_match = re.search(r'[Tt]otal\s*=?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)', text, re.I)
                if total_match:
                    sqft_str = total_match.group(1).replace(',', '')
                    sqft = int(float(sqft_str))
                    if 100 <= sqft <= 100000:
                        sqm = self._extract_sqm(text)
                        return sqft, sqm
            except Exception:
                continue

        # Step 3: FALLBACK - Find ALL sqft values and take the largest reasonable one
        # This catches cases where "Total" isn't explicitly mentioned
        all_matches = re.findall(sqft_pattern, full_text, re.I)

        if all_matches:
            candidates = []
            for m in all_matches:
                try:
                    val_str = m.replace(',', '')
                    val = float(val_str)
                    # Filter reasonable property size range (exclude tiny room measurements)
                    # Minimum 200 sqft for total area to avoid catching room dimensions
                    if 200 < val < 100000:
                        candidates.append(int(val))
                except ValueError:
                    continue

            if candidates:
                # Take the largest value as it's most likely the total
                sqft = max(candidates)
                sqm = self._extract_sqm(full_text)
                return sqft, sqm

        return None, None

    def _extract_sqm(self, text: str) -> Optional[int]:
        """Extract sqm value from text."""
        sqm_pattern = r'(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*m|sqm|Sq\s*M|square\s*m)'
        sqm_match = re.search(sqm_pattern, text, re.I)
        if sqm_match:
            try:
                return int(float(sqm_match.group(1).replace(',', '')))
            except (ValueError, IndexError):
                pass
        return None

    def _extract_address(self, text: str) -> Optional[str]:
        """Extract property address from floorplan text."""
        # Look for patterns like "123 Street Name, SW1" or "Property Name, Area"
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            # Look for postcode pattern
            if re.search(r'[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}', line):
                return line
            # Or street name patterns
            if re.search(r'(?:Street|Road|Avenue|Lane|Gardens|Square|Place|Terrace)', line, re.I):
                return line
        return None

    def _extract_rooms(self, text: str) -> List[Room]:
        """Extract individual rooms with dimensions."""
        rooms = []

        # Pattern for room with dimensions
        # e.g., "Principal Bedroom 19'9 x 13'7 (6.02 x 4.14)" or "Kitchen 5.16 x 3.35 16'11 x 11'0"

        # Imperial dimensions pattern: 19'9" x 13'7" or 19'9 x 13'7
        imperial_pattern = r"(\d+)['\u2019](\d+)?[\"″]?\s*[x×]\s*(\d+)['\u2019](\d+)?[\"″]?"

        # Metric dimensions pattern: 6.02 x 4.14 or 6.02m x 4.14m
        metric_pattern = r"(\d+\.?\d*)\s*m?\s*[x×]\s*(\d+\.?\d*)\s*m?"

        # Split into lines and process
        lines = text.split('\n')

        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                continue

            # Try to identify room type
            room_type = None
            room_name = None

            for rtype, pattern in self.ROOM_TYPES.items():
                match = re.search(pattern, line_lower)
                if match:
                    room_type = rtype
                    # Extract full room name from original line
                    room_name = self._extract_room_name(line, match.start(), match.end())
                    break

            if room_type:
                room = Room(type=room_type, name=room_name or room_type.title())

                # Extract imperial dimensions
                imp_match = re.search(imperial_pattern, line)
                if imp_match:
                    room.dimensions_imperial = imp_match.group(0)
                    # Calculate approximate sqft
                    try:
                        ft1 = int(imp_match.group(1)) + (int(imp_match.group(2) or 0) / 12)
                        ft2 = int(imp_match.group(3)) + (int(imp_match.group(4) or 0) / 12)
                        room.sqft = int(ft1 * ft2)
                    except:
                        pass

                # Extract metric dimensions
                met_match = re.search(metric_pattern, line)
                if met_match:
                    room.dimensions_metric = f"{met_match.group(1)} x {met_match.group(2)}"
                    try:
                        m1 = float(met_match.group(1))
                        m2 = float(met_match.group(2))
                        room.sqm = round(m1 * m2, 1)
                    except:
                        pass

                # Extract ceiling height (CH 2.54m or CH 8'6)
                ch_match = re.search(r'CH\s*(\d+\.?\d*)\s*m?', line, re.I)
                if ch_match:
                    room.ceiling_height = ch_match.group(1) + 'm'

                rooms.append(room)

        return rooms

    def _extract_room_name(self, line: str, start: int, end: int) -> str:
        """Extract the full room name from a line."""
        # Look for capitalized words before or around the match
        words = []
        for word in line.split():
            if word[0].isupper() or word.lower() in ['and', 'with', 'room']:
                words.append(word)
            elif words:
                break
        return ' '.join(words) if words else None

    def _count_rooms(self, rooms: List[Room]) -> Dict[str, int]:
        """Count rooms by type."""
        counts = {}
        for room in rooms:
            counts[room.type] = counts.get(room.type, 0) + 1
        return counts

    def _extract_floors(self, text: str) -> List[str]:
        """Extract floor level names and return display names."""
        # Apply text normalization and exclusion filtering
        normalized_text = self._normalize_text(text)
        filtered_text = self._filter_exclusion_lines(normalized_text)

        floors_found = set()
        text_lower = filtered_text.lower()

        for pattern, (canonical, display_name) in self.FLOOR_PATTERNS.items():
            if re.search(pattern, text_lower):
                floors_found.add(display_name)

        return sorted(list(floors_found))

    def _classify_floors(self, text: str) -> FloorData:
        """Extract floors and return binary classification for ML training."""
        floor_data = FloorData()

        # Apply text normalization and exclusion filtering
        normalized_text = self._normalize_text(text)
        filtered_text = self._filter_exclusion_lines(normalized_text)
        text_lower = filtered_text.lower()

        # Track which canonical floors are found
        canonical_floors = set()

        for pattern, (canonical, display_name) in self.FLOOR_PATTERNS.items():
            if re.search(pattern, text_lower):
                canonical_floors.add(canonical)
                floor_data.floors_raw.append(display_name)

        # Set binary flags based on canonical floor names
        if 'basement' in canonical_floors:
            floor_data.has_basement = 1

        if 'lower_ground' in canonical_floors:
            floor_data.has_lower_ground = 1

        if 'ground' in canonical_floors:
            floor_data.has_ground = 1

        if 'mezzanine' in canonical_floors:
            floor_data.has_mezzanine = 1

        if 'first' in canonical_floors:
            floor_data.has_first_floor = 1

        if 'second' in canonical_floors:
            floor_data.has_second_floor = 1

        if 'third' in canonical_floors:
            floor_data.has_third_floor = 1

        # Fourth floor and above
        if any(f in canonical_floors for f in ['fourth', 'fifth', 'sixth', 'seventh', 'penthouse']):
            floor_data.has_fourth_plus = 1

        if 'roof_terrace' in canonical_floors:
            floor_data.has_roof_terrace = 1

        # Count unique floors (mezzanine counted as a floor, roof_terrace not counted)
        main_floors = {'basement', 'lower_ground', 'ground', 'mezzanine', 'first', 'second', 'third',
                       'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
                       'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth',
                       'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth'}
        floor_data.floor_count = len(canonical_floors & main_floors)

        # Classify property level
        if floor_data.floor_count <= 1:
            floor_data.property_levels = 'single_floor'
        elif floor_data.floor_count == 2:
            floor_data.property_levels = 'duplex'
        elif floor_data.floor_count == 3:
            floor_data.property_levels = 'triplex'
        else:
            floor_data.property_levels = 'multi_floor'

        return floor_data

    def _extract_outdoor_spaces(self, text: str) -> List[OutdoorSpace]:
        """Extract outdoor spaces like terraces, gardens, balconies."""
        spaces = []
        text_lower = text.lower()

        # Imperial dimensions pattern
        imperial_pattern = r"(\d+)['\u2019](\d+)?[\"″]?\s*[x×]\s*(\d+)['\u2019](\d+)?[\"″]?"

        for space_type, pattern in self.OUTDOOR_TYPES.items():
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                space = OutdoorSpace(type=space_type)

                # Look for dimensions near the match
                context_start = max(0, match.start() - 10)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]

                dim_match = re.search(imperial_pattern, context)
                if dim_match:
                    space.dimensions_imperial = dim_match.group(0)
                    try:
                        ft1 = int(dim_match.group(1)) + (int(dim_match.group(2) or 0) / 12)
                        ft2 = int(dim_match.group(3)) + (int(dim_match.group(4) or 0) / 12)
                        space.sqft = int(ft1 * ft2)
                    except:
                        pass

                spaces.append(space)

        return spaces

    def _extract_special_features(self, text: str) -> List[str]:
        """Extract special features mentioned in the floorplan."""
        features = []
        text_lower = text.lower()

        for feature in self.SPECIAL_FEATURES:
            if feature in text_lower:
                features.append(feature)

        return list(set(features))  # Dedupe

    def _calculate_confidence(self, data: FloorplanData) -> float:
        """Calculate confidence score based on extraction quality."""
        score = 0.0
        max_score = 0.0

        # Total area (high value)
        max_score += 0.4
        if data.total_sqft:
            score += 0.4

        # Rooms extracted
        max_score += 0.3
        if data.rooms:
            room_score = min(len(data.rooms) / 5.0, 1.0) * 0.3  # Up to 5 rooms = full score
            score += room_score

        # Room dimensions
        max_score += 0.2
        rooms_with_dims = sum(1 for r in data.rooms if r.dimensions_imperial or r.dimensions_metric)
        if data.rooms:
            score += (rooms_with_dims / len(data.rooms)) * 0.2

        # Floors identified
        max_score += 0.1
        if data.floors:
            score += 0.1

        return round(score / max_score, 2) if max_score > 0 else 0.0


# Convenience function
def extract_from_floorplan(image_source) -> Dict[str, Any]:
    """
    Convenience function to extract data from a floorplan.

    Args:
        image_source: File path (str), bytes, or PIL Image

    Returns:
        Dictionary with extracted data
    """
    extractor = FloorplanExtractor()

    if isinstance(image_source, str):
        data = extractor.extract_from_file(image_source)
    elif isinstance(image_source, bytes):
        data = extractor.extract_from_bytes(image_source)
    elif isinstance(image_source, Image.Image):
        data = extractor.extract_from_image(image_source)
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    return data.to_dict()
