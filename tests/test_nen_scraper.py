import os
import unittest

from jnd_utils.log import init_logging

from src.documents_parser import DocumentsParser
from src.scrapers.nen_scraper import NenScraper, NenContractDetail
from tests.mocks.mock_database import TestDatabaseManager
from tests.mocks.mock_http_client import MockHttpClient


class TestNenScraper(unittest.TestCase):
    """Test the NEN Parser with real test database."""
    TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    @classmethod
    def setUpClass(cls):
        """Set up test resources once for all tests."""
        init_logging()
        cls.test_data_path = os.path.join(cls.TEST_DATA_PATH, "nen_mock_html")
        os.makedirs(cls.test_data_path, exist_ok=True)

        # Create test database ONCE for entire test class
        cls.db_manager = TestDatabaseManager()
        cls.test_database = cls.db_manager.create_test_database()
        cls.mock_client = MockHttpClient(cls.test_data_path)
        print(f"Test database created: {os.environ.get('DB_NAME')}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if hasattr(cls, 'db_manager'):
            cls.db_manager.cleanup()
        print("Test database cleaned up")

    def setUp(self):
        """Set up resources for each test."""
        # Use the shared test database instance
        self.parser = NenScraper(
            date_from="2025-05-06",
            deadline="2025-05-06",
            http_client=self.mock_client,
            database=self.test_database  # Use class-level database
        )

    def tearDown(self):
        """Clean up database after each test."""
        if hasattr(self, 'db_manager'):
            self.db_manager.cleanup()

    def test_get_all_items_on_page(self):
        """Test parsing multiple contracts from a listing page."""
        items = self.parser.get_all_items_on_page()
        self.assertEqual(len(items), 50)

        # Verify first item details
        first_item = items[0]
        self.assertEqual(first_item['system_number'], 'N006/25/V00011701')
        self.assertEqual(first_item['title'],
                         'KŘP KvK - NPO - Sokolov, Jednoty 1773 - energeticky úsporná opatření, reg. č. 4181000033')
        self.assertEqual(first_item['status'], 'Not terminated')
        self.assertEqual(first_item['contracting_authority'], 'Krajské ředitelství policie Karlovarského kraje')
        self.assertEqual(first_item['deadline'], '05/09/2025, 10:00 AM')

        # Verify last item details
        last_item = items[-1]
        self.assertEqual(last_item['system_number'], 'N006/25/V00015367')
        self.assertEqual(last_item['title'], 'Mediální poradenství pro CzechTourism')
        self.assertEqual(last_item['status'], 'Not terminated')
        self.assertEqual(last_item['contracting_authority'], 'Česká centrála cestovního ruchu - CzechTourism')
        self.assertEqual(last_item['deadline'], '05/27/2025, 09:00 AM')

    def test_parse_contract_detail(self):
        """Test parsing contract detail and saving to real database."""
        test_item = {
            'system_number': 'N006/25/V00015462',
            'title': 'Modul Cisco - OB7125-044',
            'status': 'Not terminated',
            'contracting_authority': 'Ministerstvo zahraničních věcí',
            'deadline': '15. 05. 2025 09:30',
            'detail_url': 'https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015462',
            'detail_url_short': 'https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015462',
        }

        # Parse contract detail
        contract_detail = self.parser.scrape_contract_detail(test_item)
        self.assertIsInstance(contract_detail, NenContractDetail)

        # Basic Information assertions
        self.assertEqual(contract_detail.nen_system_number, 'N006/25/V00015462')
        self.assertEqual(contract_detail.procurement_procedure_name, 'Modul Cisco - OB7125-044')
        self.assertEqual(contract_detail.contracting_authority, 'Ministerstvo zahraničních věcí')
        self.assertEqual(contract_detail.current_status_of_the_procurement_procedure, 'Not terminated')
        self.assertEqual(contract_detail.division_into_lots, 'No')
        self.assertEqual(contract_detail.procurement_procedure_type, 'Otevřená výzva')
        self.assertEqual(contract_detail.specifications_of_the_procurement_procedure, 'otevřená výzva')
        self.assertEqual(contract_detail.type, 'Public supply contract')

        # Test saving to real database
        self.parser.save_contract_detail(contract_detail)

        # contract_detail_filtered = contract_detail.filter_for_llm_parsing()

        # Uncomment when ready to test LLM extraction
        # extractor = TenderExtractor()
        # extracted = extractor.process(doc_data, contract_detail_filtered)
        # print(extracted)


if __name__ == "__main__":
    init_logging()
    unittest.main()
