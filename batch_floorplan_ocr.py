#!/usr/bin/env python3
"""
Batch Floorplan OCR Analysis

Processes all floorplan URLs in the database through the FloorplanExtractor
to extract floor-level data, room details, and sqft (if missing).

Usage:
    python3 batch_floorplan_ocr.py                    # Process all
    python3 batch_floorplan_ocr.py --limit 100       # Limit to 100
    python3 batch_floorplan_ocr.py --source rightmove # Single source
    python3 batch_floorplan_ocr.py --dry-run         # Test without DB writes
"""

import sqlite3
import requests
import time
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# Import the optimized FloorplanExtractor
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'property_scraper' / 'utils'))
from floorplan_extractor import FloorplanExtractor, FloorplanData


class BatchFloorplanOCR:
    """Batch process floorplan images through OCR extraction."""

    def __init__(self, db_path: str = 'output/rentals.db', dry_run: bool = False):
        self.db_path = db_path
        self.dry_run = dry_run
        self.extractor = FloorplanExtractor()

        # Stats
        self.stats = {
            'total': 0,
            'processed': 0,
            'success': 0,
            'sqft_extracted': 0,
            'floors_extracted': 0,
            'failed_download': 0,
            'failed_ocr': 0,
            'start_time': time.time(),
        }

    def get_properties_to_process(self, source: Optional[str] = None,
                                   limit: Optional[int] = None) -> list:
        """Get properties with floorplan URLs that need OCR analysis."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get properties with floorplan_url but missing floor data
        query = '''
            SELECT id, source, property_id, address, floorplan_url, size_sqft
            FROM listings
            WHERE floorplan_url IS NOT NULL
            AND floorplan_url != ''
            AND (floor_count IS NULL OR floor_count = 0)
        '''
        params = []

        if source:
            query += ' AND source = ?'
            params.append(source)

        if limit:
            query += f' LIMIT {limit}'

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def download_image(self, url: str, timeout: int = 15) -> Optional[bytes]:
        """Download floorplan image from URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            return None

    def process_property(self, prop: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single property through OCR."""
        result = {
            'id': prop['id'],
            'property_id': prop['property_id'],
            'source': prop['source'],
            'success': False,
            'data': None,
            'error': None,
        }

        # Download image
        image_bytes = self.download_image(prop['floorplan_url'])
        if not image_bytes:
            result['error'] = 'download_failed'
            return result

        # Run OCR extraction
        try:
            data = self.extractor.extract_from_bytes(image_bytes)
            result['success'] = True
            result['data'] = data
        except Exception as e:
            result['error'] = f'ocr_failed: {str(e)[:50]}'

        return result

    def update_database(self, prop_id: int, data: FloorplanData, existing_sqft: Optional[int]):
        """Update database with extracted data."""
        if self.dry_run:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build update fields
        updates = []
        params = []

        # Floor data
        if data.floor_data:
            fd = data.floor_data
            updates.extend([
                'has_basement = ?',
                'has_lower_ground = ?',
                'has_ground = ?',
                'has_mezzanine = ?',
                'has_first_floor = ?',
                'has_second_floor = ?',
                'has_third_floor = ?',
                'has_fourth_plus = ?',
                'has_roof_terrace = ?',
                'floor_count = ?',
                'property_levels = ?',
            ])
            params.extend([
                fd.has_basement,
                fd.has_lower_ground,
                fd.has_ground,
                fd.has_mezzanine,
                fd.has_first_floor,
                fd.has_second_floor,
                fd.has_third_floor,
                fd.has_fourth_plus,
                fd.has_roof_terrace,
                fd.floor_count,
                fd.property_levels,
            ])

        # Room details (JSON)
        if data.rooms:
            room_details = json.dumps([{
                'type': r.type,
                'name': r.name,
                'dimensions': r.dimensions_imperial or r.dimensions_metric,
                'sqft': r.sqft,
            } for r in data.rooms])
            updates.append('room_details = ?')
            params.append(room_details)

        # Sqft (only update if currently missing)
        if data.total_sqft and not existing_sqft:
            updates.append('size_sqft = ?')
            params.append(data.total_sqft)

        if updates:
            params.append(prop_id)
            query = f"UPDATE listings SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()

        conn.close()

    def run(self, source: Optional[str] = None, limit: Optional[int] = None,
            workers: int = 4):
        """Run batch OCR processing."""
        print("=" * 70)
        print("BATCH FLOORPLAN OCR ANALYSIS")
        print("=" * 70)
        print(f"[CONFIG] Database: {self.db_path}")
        print(f"[CONFIG] Dry run: {self.dry_run}")
        print(f"[CONFIG] Workers: {workers}")
        if source:
            print(f"[CONFIG] Source filter: {source}")
        if limit:
            print(f"[CONFIG] Limit: {limit}")
        print()

        # Get properties to process
        properties = self.get_properties_to_process(source, limit)
        self.stats['total'] = len(properties)

        print(f"[START] Found {len(properties)} properties to process")
        print()

        if not properties:
            print("[DONE] No properties to process")
            return

        # Process with thread pool for parallel downloads
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.process_property, p): p for p in properties}

            for future in as_completed(futures):
                prop = futures[future]
                self.stats['processed'] += 1

                try:
                    result = future.result()

                    if result['success']:
                        self.stats['success'] += 1
                        data = result['data']

                        # Track what we extracted
                        if data.floor_data and data.floor_data.floor_count > 0:
                            self.stats['floors_extracted'] += 1
                        if data.total_sqft and not prop.get('size_sqft'):
                            self.stats['sqft_extracted'] += 1

                        # Update database
                        self.update_database(prop['id'], data, prop.get('size_sqft'))

                    else:
                        if result['error'] == 'download_failed':
                            self.stats['failed_download'] += 1
                        else:
                            self.stats['failed_ocr'] += 1

                except Exception as e:
                    self.stats['failed_ocr'] += 1

                # Progress logging
                if self.stats['processed'] % 25 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    rate = self.stats['processed'] / elapsed if elapsed > 0 else 0
                    print(f"[PROGRESS] {self.stats['processed']}/{self.stats['total']} | "
                          f"Success: {self.stats['success']} | "
                          f"Floors: {self.stats['floors_extracted']} | "
                          f"Rate: {rate:.1f}/s")

        # Final summary
        self._print_summary()

    def _print_summary(self):
        """Print final summary."""
        elapsed = time.time() - self.stats['start_time']

        print()
        print("=" * 70)
        print("BATCH OCR COMPLETE")
        print("=" * 70)
        print(f"[SUMMARY] Duration: {elapsed:.1f}s")
        print(f"[SUMMARY] Properties processed: {self.stats['processed']}")
        print(f"[SUMMARY] OCR successful: {self.stats['success']} ({self.stats['success']/max(self.stats['processed'],1)*100:.0f}%)")
        print(f"[SUMMARY] Floors extracted: {self.stats['floors_extracted']}")
        print(f"[SUMMARY] Sqft extracted: {self.stats['sqft_extracted']}")
        print(f"[SUMMARY] Download failures: {self.stats['failed_download']}")
        print(f"[SUMMARY] OCR failures: {self.stats['failed_ocr']}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Batch floorplan OCR analysis')
    parser.add_argument('--source', type=str, help='Filter by source (rightmove, foxtons, etc.)')
    parser.add_argument('--limit', type=int, help='Limit number of properties to process')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--dry-run', action='store_true', help='Test without writing to database')
    parser.add_argument('--db', type=str, default='output/rentals.db', help='Database path')

    args = parser.parse_args()

    processor = BatchFloorplanOCR(db_path=args.db, dry_run=args.dry_run)
    processor.run(source=args.source, limit=args.limit, workers=args.workers)


if __name__ == '__main__':
    main()
