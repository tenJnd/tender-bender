# tests/test_documents_parser.py
import os
import unittest

from jnd_utils.log import init_logging

from src.documents_parser import DocumentsParser, ParsedDocumentData
from tests.mocks.mock_http_client import MockHttpClient


class TestDocumentsParser(unittest.TestCase):
    TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    def setUp(self):
        init_logging()
        self.mock_client = MockHttpClient(self.TEST_DATA_PATH)

        # Define test document info for different file types
        self.pdf_doc_info = [
            {
                "file": "test_pdf_file.pdf",
                "id": "2865511219",
                "download_link": "https://nen.nipez.cz/file?id=2865511219"
            }
        ]

        self.docx_doc_info = [
            {
                "file": "test_docx_file.docx",
                "id": "2865511217",
                "download_link": "https://nen.nipez.cz/file?id=2865511217"
            }
        ]

        self.zip_doc_info = [
            {
                "file": "test_zip_file.zip",
                "id": "2865511218",
                "download_link": "https://nen.nipez.cz/file?id=2865511218"
            }
        ]

    def test_parse_pdf_document(self):
        """Test parsing a PDF document."""
        # Create a parser for PDF document
        pdf_parser = DocumentsParser(self.pdf_doc_info, http_client=self.mock_client)

        # Get processed documents
        documents_data = pdf_parser.get_documents_data()

        # Assertions
        self.assertEqual(len(documents_data), 1)
        doc = documents_data[0]
        self.assertEqual(doc.id, "2865511219")
        self.assertEqual(doc.name, "test_pdf_file.pdf")
        self.assertEqual(doc.type, "pdf")
        self.assertTrue(os.path.exists(doc.path))
        self.assertTrue(len(doc.preview) > 0)
        self.assertTrue(len(doc.full_text) > 0)

    def test_parse_docx_document(self):
        """Test parsing a DOCX document."""
        # Create a parser for DOCX document
        docx_parser = DocumentsParser(self.docx_doc_info, http_client=self.mock_client)

        # Get processed documents
        documents_data = docx_parser.get_documents_data()

        # Assertions
        self.assertEqual(len(documents_data), 1)
        doc = documents_data[0]
        self.assertEqual(doc.id, "2865511217")
        self.assertEqual(doc.name, "test_docx_file.docx")
        self.assertEqual(doc.type, "docx")
        self.assertTrue(os.path.exists(doc.path))
        self.assertTrue(len(doc.preview) > 0)
        self.assertTrue(len(doc.full_text) > 0)

    def test_parse_zip_document(self):
        """Test parsing a ZIP document containing multiple files."""
        # Create a parser for ZIP document
        # Note: This test assumes the zip file contains at least one supported document
        zip_parser = DocumentsParser(self.zip_doc_info, http_client=self.mock_client)

        # Get processed documents
        documents_data = zip_parser.get_documents_data()

        # Assertions
        # We expect at least one document to be extracted from the ZIP
        self.assertGreater(len(documents_data), 1)

        # Check that all extracted documents have valid data
        for doc in documents_data:
            self.assertTrue(doc.id.startswith("2865511218_"))  # ZIP contents should have IDs like "3_1", "3_2", etc.
            self.assertTrue(doc.name)  # Should have a name
            self.assertTrue(doc.path)  # Should have a path
            self.assertTrue(os.path.exists(doc.path))  # File should exist

    def test_get_full_data(self):
        """Test retrieving full data for a document by name."""
        pdf_parser = DocumentsParser(self.pdf_doc_info, http_client=self.mock_client)

        # Get document by name
        doc = pdf_parser.get_full_data("test_pdf_file.pdf")

        # Assertions
        self.assertIsInstance(doc, ParsedDocumentData)
        self.assertEqual(doc.id, "2865511219")
        self.assertEqual(doc.name, "test_pdf_file.pdf")

        # Try with a non-existent document
        nonexistent_doc = pdf_parser.get_full_data("nonexistent.pdf")
        self.assertIsNone(nonexistent_doc)


if __name__ == "__main__":
    unittest.main()
