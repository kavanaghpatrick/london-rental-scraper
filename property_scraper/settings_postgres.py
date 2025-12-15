# Settings for Vercel Postgres deployment
# Use this for GitHub Actions when writing to cloud Postgres

from property_scraper.settings import *

# Override pipelines to use Postgres
ITEM_PIPELINES = {
    'property_scraper.pipelines.CleanDataPipeline': 100,
    'property_scraper.pipelines.DuplicateFilterPipeline': 200,
    'property_scraper.pipelines.JsonWriterPipeline': 300,
    'property_scraper.pipelines_postgres.PostgresPipeline': 400,  # Postgres instead of SQLite
}

# Override extensions to use Postgres audit logger
EXTENSIONS = {
    'property_scraper.extensions.audit_logger_postgres.PostgresAuditLoggerExtension': 100,
}

# For standard (non-Playwright) spiders
# SCRAPY_SETTINGS_MODULE=property_scraper.settings_postgres_standard
