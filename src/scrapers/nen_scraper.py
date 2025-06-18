import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from database_tools.adapters.postgresql import PostgresqlAdapter
from jnd_utils.log import init_logging
from tornado.httpclient import HTTPClient

from src.scrapers.base_scraper import BaseScraper
from src.utils.helpers import filter_dataclass
from tests.mocks.mock_http_client import MockHttpClient

logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://nen.nipez.cz"
LISTING_PATH = "/en/verejne-zakazky"
DETAIL_PATH = "/detail-zakazky"
DOCUMENTS_PATH = "/zadavaci-dokumentace"
LOTS_PATH = "/casti"

WHOLE_DETAIL_PATH = f"{BASE_URL}{LISTING_PATH}{DETAIL_PATH}"


@dataclass
class NenContractDetail:
    """Class representing details of a contract from NEN system."""
    procurement_procedure_name: Optional[str] = None
    contracting_authority: Optional[str] = None
    nen_system_number: Optional[str] = None
    ien_system_number: Optional[str] = None
    contract_registration_number_in_the_vvz: Optional[str] = None
    current_status_of_the_procurement_procedure: Optional[str] = None
    division_into_lots: Optional[str] = None
    procurement_procedure_id_on_the_contracting_authoritys_profile: Optional[str] = None
    public_contract_regime: Optional[str] = None
    procurement_procedure_type: Optional[str] = None
    specifications_of_the_procurement_procedure: Optional[str] = None
    type: Optional[str] = None
    estimated_value_excl_vat: Optional[str] = None
    currency: Optional[str] = None
    date_of_publication_on_profile: Optional[str] = None
    deadline_for_submitting_tenders: Optional[str] = None
    name: Optional[str] = None
    surname: Optional[str] = None
    email: Optional[str] = None
    phone_1: Optional[str] = None
    subject_matter_description: Optional[str] = None
    code_from_the_nipez_code_list: Optional[str] = None
    name_from_the_nipez_code_list: Optional[str] = None
    main_place_of_performance: Optional[str] = None
    code_from_the_cpv_code_list: Optional[str] = None
    name_from_the_cpv_code_list: Optional[str] = None
    subject_matter_name: Optional[str] = None
    text_field_for_describing_the_place_of_performance: Optional[str] = None
    awarded_on_the_basis_of_a_framework_agreement: Optional[str] = None
    awarded_in_a_dns: Optional[str] = None
    the_result_of_the_pp_will_be_the_implementation_of_a_dns: Optional[str] = None
    this_is_a_framework_agreement: Optional[str] = None
    imported_public_contract: Optional[str] = None
    publication_records: List[Dict[str, str]] = field(default_factory=list)
    subject_matter_items: List[Dict[str, str]] = field(default_factory=list)
    place_of_performance: Dict[str, str] = field(default_factory=dict)
    detail_url: Optional[str] = None
    documents: Dict[str, str] = field(default_factory=dict)
    unmapped: Dict[str, str] = field(default_factory=dict)

    def to_dict(self, fields: List[str] = None) -> Dict[str, Any]:
        """
        Convert the contract to a dictionary with only the specified fields.
        If fields is None, returns all fields.
        """
        if fields is None:
            return asdict(self)

        result = {}
        for field in fields:
            if hasattr(self, field):
                result[field] = getattr(self, field)
        return result

    def filter_for_llm_parsing(self) -> Dict[str, Any]:
        """Return a simplified dictionary suitable for JSON serialization."""
        return filter_dataclass(
            self,
            exclude=[
                "documents", "unmapped", "name", "surname", "email", "phone_1", "detail_url",
                "detail_url", "ien_system_number", "contract_registration_number_in_the_vvz",
                "code_from_the_nipez_code_list", "name_from_the_nipez_code_list", "publication_records",
            ]
        )


class NenScraper(BaseScraper):
    """Parser for NEN (National Electronic Tool) procurement data."""

    def __init__(self,
                 date_from: str,
                 deadline: str,
                 http_client: Optional[HTTPClient | MockHttpClient] = None,
                 database: PostgresqlAdapter = None):
        super().__init__(http_client=http_client, database=database)

        logger.info(f"Initializing NenParser with date_from={date_from} and deadline={deadline}")
        self.params = (
            f"p:vz:stavZP=neukoncena"
            f"&podaniLhuta={deadline}%2C"
            f"&datumPrvniUver={date_from}%2C"
        )

    def _get_url(self, page_range: str = None) -> str:
        """Generate URL for listing page with optional pagination."""
        url = f"{BASE_URL}{LISTING_PATH}/{self.params}"
        if page_range:
            url += f"&page={page_range}"
        logger.debug(f"Generated URL: {url}")
        return url

    def get_total_pages(self) -> int:
        """Get total number of pages in search results."""
        logger.info("Determining total number of pages in search results")
        soup = self.http_client.get_soup(self._get_url())
        if soup is None:
            logger.warning("Could not fetch search results page, defaulting to 1 page")
            return 1

        pagination_holder = soup.select_one("div.gov-pagination__holder")
        if not pagination_holder:
            logger.info("No pagination found, assuming single page of results")
            return 1

        pagination_items = pagination_holder.select(
            "a.gov-pagination__item:not(.gov-pagination__item--arrow-left):not(.gov-pagination__item--arrow-right)"
        )

        page_numbers = [
            int(m.group(1)) for item in pagination_items
            if (href := item.get("href")) and (m := re.search(r'page=(\d+)', href))
        ]

        total_pages = max(page_numbers) if page_numbers else 1
        logger.info(f"Found {total_pages} page(s) of results")
        return total_pages

    @staticmethod
    def scrape_table(soup: BeautifulSoup) -> List[dict]:
        """Parse contract listing table."""
        logger.info("Parsing contract listing table")
        results = []
        tbody = soup.find("tbody", class_="gov-table__body")
        if not tbody:
            logger.warning("No table body found in search results")
            return results

        rows = tbody.find_all("tr", class_="gov-table__row")
        logger.info(f"Found {len(rows)} rows in contract listing table")

        for i, row in enumerate(rows, 1):
            cells = row.find_all("td", class_="gov-table__cell")
            if len(cells) < 6:
                logger.warning(f"Row {i} has insufficient cells ({len(cells)}), skipping")
                continue

            href_tag = row.find("a", class_="gov-link", href=True)
            detail_url = urljoin(BASE_URL, href_tag["href"]) if href_tag else None

            system_number = cells[1].get_text(strip=True)
            title = cells[2].get_text(strip=True)

            detail_url_short = (f"{WHOLE_DETAIL_PATH}/"
                                f"{system_number.replace('/', '-')}") if system_number else None

            logger.debug(f"Row {i}: Found contract {system_number} - '{title}'")

            results.append({
                "system_number": system_number,
                "title": title,
                "status": cells[3].get_text(strip=True),
                "contracting_authority": cells[4].get_text(strip=True),
                "deadline": cells[5].get_text(strip=True),
                "detail_url": detail_url,
                "detail_url_short": detail_url_short
            })

        logger.info(f"Successfully parsed {len(results)} contracts from table")
        return results

    def get_all_items_on_page(self) -> List[dict]:
        """Get all contract items from all pages."""
        total_pages = self.get_total_pages()
        page_range = f"1-{total_pages}"
        logger.info(f"Scraping combined page range: {page_range}")

        soup = self.http_client.get_soup(self._get_url(page_range))
        if not soup:
            logger.error("Failed to retrieve listing page, returning empty list")
            return []

        items = self.scrape_table(soup)
        logger.info(f"Retrieved {len(items)} total items from {total_pages} page(s)")
        return items

    @staticmethod
    def scrape_table_block(block: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse a specific table block within a detail page."""
        table = block.select_one("table.gov-table")
        if not table:
            logger.debug("No table found in content block")
            return []

        headers = [th.get_text(strip=True).lower().replace(' ', '_')
                   for th in table.select("thead th")
                   if th.get_text(strip=True)]

        logger.debug(f"Found table with headers: {headers}")

        result = []
        rows = table.select("tbody tr")
        logger.debug(f"Found {len(rows)} rows in table")

        for i, row in enumerate(rows, 1):
            cells = row.select("td")
            row_data = {headers[i]: cells[i].get_text(strip=True)
                        for i in range(len(headers)) if i < len(cells)}

            link = row.select_one("a.gov-link")
            if link:
                row_data["details_link"] = urljoin(BASE_URL, link["href"])
                row_data["id"] = row_data['details_link'].split('/')[-1]
                logger.debug(f"Row {i}: Found link with ID {row_data['id']}")

            result.append(row_data)

        logger.debug(f"Parsed {len(result)} rows from table block")
        return result

    def scrape_documents_info(self, detail_url: str) -> Dict[str, Dict[str, str]]:
        """Parse documents from contract detail page."""
        logger.info(f"Parsing documents for contract at {detail_url}")
        documents_url = f"{detail_url}{DOCUMENTS_PATH}"
        soup = self.http_client.get_soup(documents_url)

        if not soup:
            logger.warning(f"Failed to load documents page for {detail_url}")
            return {}

        for block in soup.select("div.gov-content-block"):
            section = block.select_one("h2")
            section_name = section.get_text(strip=True) if section else ""
            logger.debug(f"Processing content block: '{section_name}'")

            if section_name == "Procurement Documentation":
                docs = self.scrape_table_block(block)
                logger.info(f"Found {len(docs)} documents in Procurement Documentation section")

                for doc in docs:
                    doc_id = doc['id']
                    doc['download_link'] = f"{BASE_URL}/file?id={doc_id}"
                return docs

        logger.warning("No procurement documentation section found")
        return {}

    def scrape_contract_detail(self, item: dict) -> NenContractDetail:
        """Parse detailed contract information from detail page."""
        system_number = item['system_number']
        logger.info(f"Parsing detailed contract information for {system_number}")
        #
        # system_number_url = system_number.replace("/", "-")
        # detail_url = f"{BASE_URL}{LISTING_PATH}{DETAIL_PATH}/{system_number_url}"
        # logger.debug(f"Detail URL: {detail_url}")
        detail_url = item['detail_url_short']

        soup = self.http_client.get_soup(detail_url)
        if not soup:
            logger.warning(f"Failed to load detail page for {detail_url}")
            return NenContractDetail(**item)

        raw_fields = {}
        publication_records = []
        subject_matter_items = []
        place_of_performance = {}

        # Parse each content block on the page
        content_blocks = soup.select("div.gov-content-block")
        logger.debug(f"Found {len(content_blocks)} content blocks on detail page")

        for i, block in enumerate(content_blocks, 1):
            section = block.select_one("h2")
            section_name = section.get_text(strip=True) if section else f"Unnamed Section {i}"
            logger.debug(f"Processing content block {i}: '{section_name}'")

            # Parse key-value pairs in tiles
            tiles = block.select(".gov-grid-tile")
            logger.debug(f"Found {len(tiles)} tiles in content block '{section_name}'")

            for tile in tiles:
                key = tile.select_one("h3.gov-title--delta")
                val = tile.select_one("p.gov-note") or tile.select_one("a.gov-link")
                if key and val:
                    key_text = key.get_text(strip=True)
                    val_text = val.get_text(strip=True)
                    raw_fields[key_text] = val_text
                    logger.debug(f"Extracted field: '{key_text}' = '{val_text}'")

            # Parse text field descriptions
            para_texts = block.select("div.gov-grid-tile--raw-text p")
            if para_texts:
                logger.debug(f"Found {len(para_texts)} paragraphs in raw text section")
                raw_fields["TEXT FIELD FOR DESCRIBING THE PLACE OF PERFORMANCE"] = para_texts[0].get_text(strip=True)

            # Parse specific sections based on headings
            if section_name == "Publication Records in the NEN System":
                publication_records = self.scrape_table_block(block)
                logger.info(f"Parsed {len(publication_records)} publication records")
            elif section_name == "Subject-Matter Items":
                subject_matter_items = self.scrape_table_block(block)
                logger.info(f"Parsed {len(subject_matter_items)} subject matter items")
            elif section_name == "Place of Performance":
                rows = self.scrape_table_block(block)
                if rows:
                    place_of_performance = rows[0]
                    logger.info(f"Parsed place of performance: {len(place_of_performance)} fields")

        # Normalize field names and create contract detail object
        logger.debug(f"Normalizing {len(raw_fields)} raw fields")
        mapped = {self._normalize_field_name(k): v for k, v in raw_fields.items()}

        logger.debug("Creating ContractDetail object")
        tender_obj = NenContractDetail(
            **{
                field.name: mapped.get(field.name)
                for field in NenContractDetail.__dataclass_fields__.values()
            }
        )

        # Set additional fields
        tender_obj.detail_url = detail_url
        tender_obj.publication_records = publication_records
        tender_obj.subject_matter_items = subject_matter_items
        tender_obj.place_of_performance = place_of_performance

        # Parse documents
        logger.info(f"Fetching documents for contract {system_number}")
        tender_obj.documents = self.scrape_documents_info(detail_url)
        logger.info(f"Found {len(tender_obj.documents)} documents")

        # Track unmapped fields for debugging
        unmapped = {k: v for k, v in mapped.items() if k not in NenContractDetail.__dataclass_fields__}
        if unmapped:
            logger.debug(f"Fields in mapped but not in dataclass: {list(unmapped.keys())}")
            tender_obj.unmapped = unmapped

        logger.info(f"Successfully parsed contract details for {system_number}")
        return tender_obj

    def scrape_detailed_items(self, already_processed: Optional[Set[str]] = None) -> List[NenContractDetail]:
        """Scrape detailed information for all contract items."""
        logger.info("Starting scrape of detailed contract items")
        already_processed = already_processed or set()
        logger.info(f"Already processed items: {len(already_processed)}")

        # Uncomment the next line for real implementation
        # base_items = self.get_all_items_on_page()

        # Comment out the mock data when using real data
        base_items = [{
            'contracting_authority': 'Lesní správa Lány', 'deadline': '05. 06. 2025 09:30',
            'detail_url': 'https://nen.nipez.cz/en/verejne-zakazky/p:vz:stavZP=neukoncena&podaniLhuta=2025-05-06,&datumPrvniUver=2025-05-06,&page=1-4/detail-zakazky/N006-25-V00015013',
            'detail_url_short': 'https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015013',
            'status': 'Not terminated', 'system_number': 'N006/25/V00015013',
            'title': 'Rekonstrukce topných médií v objektech LSL'
        }]

        logger.info(f"Found {len(base_items)} base contract items")
        detailed_items = []

        for i, item in enumerate(base_items, 1):
            system_number = item["system_number"]
            logger.info(f"Processing item {i}/{len(base_items)}: {system_number} - '{item.get('title', 'No title')}'")

            if system_number in already_processed:
                logger.info(f"Skipping already processed item: {system_number}")
                continue

            logger.info(f"Fetching detailed information for {system_number}")
            detailed = self.scrape_contract_detail(item)
            detailed_items.append(detailed)
            logger.info(f"Successfully processed {system_number}")

        logger.info(f"Scraping completed. Processed {len(detailed_items)} detailed contract items")
        return detailed_items

    def save_contract_detail(self, item: NenContractDetail):
        with self.database.session_manager() as session:
            pass
            # contract_db_obj = NenContractRaw.from_nen_contract_detail(item)
            # session.add(contract_db_obj)
            # session.commit()


if __name__ == "__main__":
    init_logging()
    logger.info("Starting NEN Scraper")
    parser = NenScraper(date_from="2025-05-06", deadline="2025-05-06")
    logger.info("Initialized parser, beginning scraping process")
    results = parser.scrape_detailed_items()
    logger.info(f"Scraping complete, retrieved {len(results)} contracts")

    for i, contract in enumerate(results, 1):
        logger.info(f"Contract {i}: {contract.nen_system_number or contract.nen_system_number} - "
                    f"{contract.procurement_procedure_name or 'No name'}")
        print(contract)

    logger.info("NEN Scraper execution complete")
