import logging

from src.utils.http_client import HttpClient
from database_tools.adapters.postgresql import PostgresqlAdapter
from src.database import tender_database

logger = logging.getLogger(__name__)


class BaseScraper:
    """Base class for web scrapers."""

    def __init__(self, http_client: HttpClient = None, database: PostgresqlAdapter = None):
        """Initialize with an optional custom HTTP client."""
        self.http_client = http_client if http_client else HttpClient()
        self.database = database if database else tender_database

    @staticmethod
    def _normalize_field_name(key: str) -> str:
        """Normalize field names for consistent mapping."""
        normalized = (key.lower().replace("'", "").replace("'", "")
                      .replace("-", "_").replace(" ", "_").replace("__", "_"))
        logger.debug(f"Normalized field name: '{key}' -> '{normalized}'")
        return normalized
