# Settings for Vercel Postgres - Standard (non-Playwright) spiders
# Use for rightmove, foxtons on GitHub Actions

from property_scraper.settings_standard import *

# Override pipelines to use Postgres
ITEM_PIPELINES = {
    'property_scraper.pipelines.CleanDataPipeline': 100,
    'property_scraper.pipelines.DuplicateFilterPipeline': 200,
    'property_scraper.pipelines.JsonWriterPipeline': 300,
    'property_scraper.pipelines_postgres.PostgresPipeline': 400,
}

# Override extensions to use Postgres audit logger
EXTENSIONS = {
    'property_scraper.extensions.audit_logger_postgres.PostgresAuditLoggerExtension': 100,
}
