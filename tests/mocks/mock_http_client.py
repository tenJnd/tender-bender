# tests/mocks/mock_http_client.py

import logging
import os
import shutil
import tempfile
from typing import Optional

from bs4 import BeautifulSoup

from src.utils.file_utils import get_file_type, get_file_extension

logger = logging.getLogger(__name__)


class MockHttpClient:
    """Mock HTTP client that reads HTML files from a test folder and handles file downloads."""

    def __init__(self, test_data_path: str):
        self.test_data_path = test_data_path
        self.request_count = 0
        self.url_to_file_mapping = {
            # Main listing URL
            "https://nen.nipez.cz/en/verejne-zakazky/p:vz:stavZP=neukoncena": "listing_page.html",
            # Detail URL
            "https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015013": "detail_page.html",
            # Documents URL
            "https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015013/zadavaci-dokumentace": "documents_page.html",
        }
        # Mapping for file downloads by file ID
        self.file_id_mappings = {
            # Example: Map a file ID to a file in the test data directory
            "2865511219": "test_pdf_file.pdf",
            "2865511217": "test_docx_file.docx",
            "2865511218": "test_zip_file.zip",
            "2865511216": "test_file_2865511216.docx"
        }
        logger.info(f"Initialized MockHttpClient with test data path: {test_data_path}")

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Load HTML from a test file based on URL mapping."""
        self.request_count += 1
        logger.info(f"Mock Request #{self.request_count}: URL: {url}")

        # Find the right test file for this URL
        file_name = None
        for url_pattern, mapped_file in self.url_to_file_mapping.items():
            if url.startswith(url_pattern):
                file_name = mapped_file
                break

        if not file_name:
            # If no exact match, try to determine the type of page
            if "/detail-zakazky/" in url and "/zadavaci-dokumentace" in url:
                file_name = "documents_page.html"
            elif "/detail-zakazky/" in url:
                file_name = "detail_page.html"
            else:
                file_name = "listing_page.html"

        file_path = os.path.join(self.test_data_path, file_name)

        if not os.path.exists(file_path):
            logger.warning(f"Mock file not found: {file_path}")
            return None

        logger.debug(f"Loading mock HTML from file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            return BeautifulSoup(html_content, "html.parser")

        except Exception as e:
            logger.error(f"Error loading mock HTML for URL {url}: {e}")
            return None

    def download_file(self, url: str, file_name: str = None) -> Optional[str]:
        """
        Mock file download by copying a file from the test data directory.
        
        Args:
            url: The URL to "download" from
            file_name: Optional file name to use for inferring file extension
            
        Returns:
            Path to the "downloaded" file or None if the mock file couldn't be found
        """
        self.request_count += 1
        logger.info(f"Mock Request #{self.request_count}: Downloading file from URL: {url}")
        
        # Extract file ID from URL if it's a file download URL
        file_id = None
        if url.startswith("https://nen.nipez.cz/file?id="):
            file_id = url.split("=")[-1]
        
        source_file = None
        if file_id and file_id in self.file_id_mappings:
            # Use the mapping if we have one for this file ID
            source_file = self.file_id_mappings[file_id]
        elif file_name:
            # Otherwise try to use the provided file name
            source_file = file_name
        else:
            # Default to a generic test file
            source_file = "test_pdf_file.pdf"
            
        source_path = os.path.join(self.test_data_path, source_file)
        
        if not os.path.exists(source_path):
            logger.warning(f"Mock file not found: {source_path}")
            return None
            
        # Create a temporary copy of the file
        file_ext = os.path.splitext(source_file)[-1]
        dest_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        dest_path = dest_file.name
        dest_file.close()
        
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Successfully copied mock file from {source_path} to {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"Error copying mock file: {e}")
            if os.path.exists(dest_path):
                os.unlink(dest_path)
            return None