import logging
import os
import tempfile
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.utils.file_utils import FileType, get_file_type, guess_file_type_from_content_type, get_file_extension

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10


class HttpClient:
    """Handles HTTP requests for HTML parsing and file downloads."""

    def __init__(self, user_agent: str = "Mozilla/5.0 (compatible; NenScraperBot/1.0)"):
        self.headers = {"User-Agent": user_agent}
        self.request_count = 0

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch HTML from URL and return BeautifulSoup object."""
        self.request_count += 1
        logger.info(f"Request #{self.request_count}: Fetching URL: {url}")

        try:
            response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            logger.debug(f"Successfully fetched URL: {url} (status code: {response.status_code})")
            return BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error fetching URL {url} (timeout: {REQUEST_TIMEOUT}s)")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching URL {url}: {e} (status code: {e.response.status_code})")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching URL {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching URL {url}: {e}")
            return None

    def download_file(self, url: str, file_name: str = None) -> Optional[str]:
        """
        Download a file from URL and save to a temporary location.
        
        Args:
            url: The URL to download from
            file_name: Optional file name to use for inferring file extension
            
        Returns:
            Path to the downloaded file or None if download failed
        """
        self.request_count += 1
        logger.info(f"Request #{self.request_count}: Downloading file from URL: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            
            # Determine file extension
            suffix = ""
            
            # Try to determine from file_name first
            if file_name:
                suffix = os.path.splitext(file_name)[-1]
                
            # If no extension from file_name, try content-disposition header
            if not suffix and response.headers.get('content-disposition'):
                import re
                cd = response.headers.get('content-disposition')
                if cd:
                    filename = re.findall('filename="(.+)"', cd)
                    if filename:
                        suffix = os.path.splitext(filename[0])[-1]
            
            # If still no extension, try content-type
            if not suffix:
                content_type = response.headers.get('content-type', '')
                file_type = guess_file_type_from_content_type(content_type)
                if file_type:
                    suffix = get_file_extension(file_type)
                else:
                    suffix = '.bin'
            
            # Create temporary file
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            logger.debug(f"Saving downloaded file to: {tmp_file.name}")
            
            # Save the content
            with open(tmp_file.name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Successfully downloaded file from {url} to {tmp_file.name}")
            return tmp_file.name
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error downloading file from {url} (timeout: {REQUEST_TIMEOUT}s)")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading file from {url}: {e} (status code: {e.response.status_code})")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error downloading file from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading file from {url}: {e}")
            return None