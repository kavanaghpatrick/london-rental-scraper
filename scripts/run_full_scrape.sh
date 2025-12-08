#!/bin/bash
#
# Full Scrape Execution Script
#
# This script:
# 1. Takes a pre-scrape snapshot
# 2. Runs all spiders
# 3. Generates a comparison report
#
# Usage:
#   ./scripts/run_full_scrape.sh           # Full scrape (all sources)
#   ./scripts/run_full_scrape.sh --quick   # Quick test (limited pages)
#   ./scripts/run_full_scrape.sh --http    # HTTP spiders only (fast)
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  LONDON RENTAL SCRAPER - FULL RUN   ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Parse arguments
QUICK_MODE=false
HTTP_ONLY=false
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        --http)
            HTTP_ONLY=true
            ;;
    esac
done

# Step 1: Pre-scrape validation
echo -e "${YELLOW}[STEP 1] Running pre-scrape validation tests...${NC}"
python3 -m pytest tests/test_scrape_validation.py -v -k "pre_scrape" --tb=short
if [ $? -ne 0 ]; then
    echo -e "${RED}Pre-scrape validation failed! Fix issues before continuing.${NC}"
    exit 1
fi
echo -e "${GREEN}Pre-scrape validation passed!${NC}"
echo ""

# Step 2: Take snapshot
echo -e "${YELLOW}[STEP 2] Taking database snapshot...${NC}"
python3 tests/test_scrape_validation.py --snapshot
echo ""

# Step 3: Run spiders
echo -e "${YELLOW}[STEP 3] Running spiders...${NC}"

# Spider configurations
if [ "$QUICK_MODE" = true ]; then
    MAX_PAGES=3
    MAX_PROPS=50
    echo -e "${YELLOW}Quick mode: max_pages=$MAX_PAGES, max_properties=$MAX_PROPS${NC}"
else
    MAX_PAGES=""
    MAX_PROPS=""
fi

# Function to run spider
run_spider() {
    local SPIDER=$1
    local SETTINGS=$2

    echo -e "${YELLOW}Running $SPIDER...${NC}"

    CMD="scrapy crawl $SPIDER"

    if [ -n "$MAX_PAGES" ]; then
        CMD="$CMD -a max_pages=$MAX_PAGES"
    fi
    if [ -n "$MAX_PROPS" ]; then
        CMD="$CMD -a max_properties=$MAX_PROPS"
    fi

    SCRAPY_SETTINGS_MODULE="$SETTINGS" $CMD 2>&1 | tee "logs/${SPIDER}_$(date +%Y%m%d_%H%M%S).log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}$SPIDER completed successfully${NC}"
    else
        echo -e "${RED}$SPIDER had errors (check log)${NC}"
    fi
    echo ""
}

mkdir -p logs

# HTTP spiders (fast)
echo -e "${YELLOW}=== HTTP Spiders ===${NC}"
run_spider "rightmove" "property_scraper.settings_standard"
run_spider "foxtons" "property_scraper.settings_standard"

if [ "$HTTP_ONLY" = false ]; then
    # Playwright spiders (slow)
    echo -e "${YELLOW}=== Playwright Spiders ===${NC}"
    run_spider "savills" "property_scraper.settings"
    run_spider "knightfrank" "property_scraper.settings"
    run_spider "chestertons" "property_scraper.settings"
fi

# Step 4: Post-scrape validation
echo -e "${YELLOW}[STEP 4] Running post-scrape validation tests...${NC}"
python3 -m pytest tests/test_scrape_validation.py -v -k "post_scrape" --tb=short
echo ""

# Step 5: Generate comparison report
echo -e "${YELLOW}[STEP 5] Generating comparison report...${NC}"
python3 tests/test_scrape_validation.py --report
echo ""

# Step 6: Data integrity check
echo -e "${YELLOW}[STEP 6] Running data integrity tests...${NC}"
python3 -m pytest tests/test_scrape_validation.py -v -k "integrity" --tb=short
echo ""

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  SCRAPE COMPLETE!                    ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Database: output/rentals.db"
echo "Logs: logs/"
echo ""
echo "To see detailed changes:"
echo "  python3 tests/test_scrape_validation.py --report"
