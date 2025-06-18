# !/usr/bin/env python3
"""
Main entry point for tender processing system
"""
from jnd_utils.log import init_logging

from src.database import tender_database
from src.scrapers.nen_scraper import NenScraper
from src.tender_pipeline import TenderProcessingPipeline
from src.vector_search.base import VectorSearch
from src.vector_search.collections import tender_indexes, company_indexes


def main():
    """Main execution function"""

    # Initialize vector search with proper collections
    vector_search = VectorSearch(
        index_definition={
            'tenders_czech': tender_indexes,
            'tenders_english': tender_indexes,
            'companies': company_indexes
        }
    )

    # Create collections
    vector_search.create_collection("tenders_czech")
    vector_search.create_collection("tenders_english")

    # Initialize pipeline
    pipeline = TenderProcessingPipeline(
        vector_search=vector_search,
        database=tender_database
    )

    # Initialize scraper
    scraper = NenScraper(
        date_from="2025-05-06",
        deadline="2025-05-06"
    )

    # Example: Process specific tender
    test_item = {
        'system_number': 'N006/25/V00015462',
        'title': 'Modul Cisco - OB7125-044',
        'status': 'Not terminated',
        'contracting_authority': 'Ministerstvo zahraničních věcí',
        'deadline': '15. 05. 2025 09:30',
        'detail_url': 'https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015462',
        'detail_url_short': 'https://nen.nipez.cz/en/verejne-zakazky/detail-zakazky/N006-25-V00015462',
    }

    # Scrape and process single tender
    try:
        # all_current_contracts = scraper.scrape_detailed_items()
        # unified_tender = pipeline.process_batch(
        #     source_name='nen',
        #     source_data_list=all_current_contracts,
        #     skip_documents=False,
        #     skip_llm=False,
        #     skip_vector_search=False
        # )

        test_item_scraped = scraper.scrape_contract_detail(test_item)
        unified_tender = pipeline.process_from_source(
            source_name="NEN",
            source_data=test_item_scraped,

        )

    except Exception as e:
        print(f"❌ Processing failed: {e}")


if __name__ == "__main__":
    init_logging()
    main()
