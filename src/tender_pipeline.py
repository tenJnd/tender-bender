import logging
from datetime import datetime
from typing import List, Optional, Any, Literal, Union

from database_tools.adapters.postgresql import PostgresqlAdapter

from src.agents.tender_llm_extractor import TenderExtractorCZ
from src.models.unified_tender import UnifiedTenderRecord, ProcessingStage
from src.processors.source_mappers import SourceMapperRegistry

logger = logging.getLogger(__name__)


class TenderProcessingPipeline:
    """
    Complete processing pipeline for tenders:
    1. Map source data to unified format
    2. Parse documents
    3. Extract semantic data with LLM
    4. Prepare for vector search
    5. Save to database
    """

    def __init__(self,
                 vector_search: Optional[Any] = None,
                 database: Optional[PostgresqlAdapter] = None):
        self.vector_search = vector_search
        self.database = database
        self.tender_extractor = None  # Initialize lazily

    def _get_tender_extractor(self, language: Literal["cz", "en"] = "cz") -> Union[TenderExtractorCZ]:
        """Get or create tender extractor"""
        if self.tender_extractor is None:
            if language == "cz":
                self.tender_extractor = TenderExtractorCZ()
            else:
                self.tender_extractor = None  # TODO add TenderExtractor for english later
        return self.tender_extractor

    def process_from_source(self,
                            source_name: str,
                            source_data: Literal["NEN", "VZZ", "TED"],
                            skip_documents: bool = False,
                            skip_llm: bool = False,
                            skip_vector_search: bool = False) -> UnifiedTenderRecord:
        """
        Process tender from any source through the complete pipeline
        
        Args:
            source_name: Name of source system (NEN, TED, etc.)
            source_data: Source-specific data (NenContractDetail, etc.)
            skip_documents: Skip document parsing
            skip_llm: Skip LLM semantic extraction
            skip_vector_search: Skip vector search indexing
            
        Returns:
            Fully processed UnifiedTenderRecord
        """
        logger.info(f"Starting pipeline processing for {source_name} tender")

        # Step 1: Map to unified format
        unified_tender = SourceMapperRegistry.map_to_unified(source_name, source_data)
        logger.info(f"Mapped {source_name} data to unified format: {unified_tender.tender_id}")

        # Step 2: Parse documents (if available and not skipped)
        if not skip_documents and unified_tender.is_documents_available:
            unified_tender = self._parse_documents(unified_tender)

        # Step 3: Extract semantic data with LLM (if not skipped)
        if not skip_llm:
            unified_tender = self._extract_semantic_data(unified_tender, source_name)

        # Step 4: Save to database
        if self.database:
            self._save_to_database(unified_tender)

        # Step 5: Index in vector search
        if self.vector_search and not skip_vector_search and unified_tender.semantic_data:
            unified_tender = self._index_in_vector_search(unified_tender)

        unified_tender.processing_stage = ProcessingStage.COMPLETED  # TODO handle this for retries
        unified_tender.processed_at = datetime.utcnow()

        logger.info(f"Completed pipeline processing for {unified_tender.tender_id}")
        return unified_tender

    @staticmethod
    def _parse_documents(tender: UnifiedTenderRecord) -> UnifiedTenderRecord:
        """Parse documents for the tender using the integrated DocumentsParser."""
        if not tender.document_infos:
            logger.warning(f"No documents to parse for tender {tender.tender_id}")
            tender.add_processing_warning("No documents available for parsing")
            return tender

        try:
            logger.info(f"Parsing {len(tender.document_infos)} documents for tender {tender.tender_id}")

            # Use the integrated class method to parse documents directly for the tender
            from src.documents_parser import DocumentsParser

            DocumentsParser.parse_documents_for_tender(tender)

            # Documents are automatically added to the tender record during parsing
            logger.info(f"Successfully parsed {len(tender.parsed_documents)} documents")

            # Update processing stage
            tender.processing_stage = ProcessingStage.DOCUMENTS_PARSED

            return tender

        except Exception as e:
            error_msg = f"Document parsing failed: {str(e)}"
            logger.error(error_msg)
            tender.add_processing_error(error_msg)
            return tender

    def _extract_semantic_data(self, tender: UnifiedTenderRecord, source_name: str) -> UnifiedTenderRecord:
        """Extract semantic data using LLM"""
        lang = 'cz' if source_name == "NEN" else 'en'

        try:
            logger.info(f"Extracting semantic data for {tender.tender_id}")

            # Prepare metadata for LLM
            metadata = tender.prepare_metadata_for_llm()

            # Use parsed documents if available, otherwise empty list
            documents = tender.parsed_documents if tender.is_documents_parsed else []

            # Extract semantic data
            extractor = self._get_tender_extractor(language=lang)
            semantic_data = extractor.process(documents, metadata)
            tender.semantic_data = semantic_data
            tender.processing_stage = ProcessingStage.SEMANTIC_PROCESSED

            logger.info(f"Extracted semantic data for {tender.tender_id}")

        except Exception as e:
            error_msg = f"Semantic extraction failed for {tender.tender_id}: {e}"
            logger.error(error_msg)
            tender.processing_errors.append(error_msg)

        return tender

    def _index_in_vector_search(self, tender: UnifiedTenderRecord) -> UnifiedTenderRecord:
        """Index tender in vector search (both Czech and English)"""
        try:
            logger.info(f"Indexing {tender.tender_id} in vector search")

            # Prepare Czech version
            czech_data = tender.prepare_data_for_vector_database()
            self.vector_search.add_items("tenders_czech", [czech_data])

            tender.processing_stage = ProcessingStage.VECTOR_INDEXED

            logger.info(f"Indexed {tender.tender_id} in vector search (both languages)")
            return tender

        except Exception as e:
            error_msg = f"Vector indexing failed for {tender.tender_id}: {e}"
            logger.error(error_msg)
            tender.processing_errors.append(error_msg)

    def _save_to_database(self, tender: UnifiedTenderRecord):
        """Save tender to database"""
        try:
            logger.info(f"Saving {tender.tender_id} to database")

            pass  # TODO: finish this

        except Exception as e:
            error_msg = f"Database save failed for {tender.tender_id}: {e}"
            logger.error(error_msg)
            tender.processing_errors.append(error_msg)

    def process_batch(self,
                      source_name: str,
                      source_data_list: List[Any],
                      batch_size: int = 10,
                      skip_documents: bool = False,
                      skip_llm: bool = False,
                      skip_vector_search: bool = False) -> List[UnifiedTenderRecord]:
        """Process multiple tenders in batches"""

        results = []
        total = len(source_data_list)

        for i in range(0, total, batch_size):
            batch = source_data_list[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

            batch_results = []
            for source_data in batch:
                try:
                    result = self.process_from_source(
                        source_name,
                        source_data,
                        skip_documents=skip_documents,
                        skip_llm=skip_llm,
                        skip_vector_search=skip_vector_search
                    )
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process tender in batch: {e}")
                    continue

            results.extend(batch_results)
            logger.info(f"Completed batch with {len(batch_results)} successful processings")

        logger.info(f"Batch processing completed: {len(results)}/{total} successful")
        return results
