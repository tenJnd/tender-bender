from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid5, NAMESPACE_DNS


class ProcessingStage(Enum):
    RAW_SCRAPED = "raw_scraped"
    UNIFIED_MAPPED = "unified_mapped"
    DOCUMENTS_PARSED = "documents_parsed"
    SEMANTIC_PROCESSED = "semantic_processed"
    VECTOR_INDEXED = "vector_indexed"
    COMPLETED = "completed"


@dataclass
class ParsedContent:
    """Represents parsed content from a document"""
    preview: str
    full_text: str


@dataclass
class ParsedDocumentData:
    """Represents a parsed document with all its metadata and content"""
    id: str
    name: str
    type: str
    path: str
    url: Optional[str] = None
    preview: str = ""
    full_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "url": self.url,
            "preview": self.preview,
            "full_text": self.full_text
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParsedDocumentData':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TenderLot:
    """Represents a lot within a tender (for multi-lot tenders)"""
    lot_id: str
    lot_number: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    cpv_code: Optional[str] = None
    cpv_description: Optional[str] = None
    estimated_value_eur: Optional[float] = None
    estimated_value_original: Optional[str] = None
    currency_original: Optional[str] = None
    location: Optional[str] = None
    items: List[Dict[str, Any]] = field(default_factory=list)  # Individual items within lot
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenderItem:
    """Represents individual items/products within a tender or lot"""
    item_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[str] = None
    unit: Optional[str] = None
    cpv_code: Optional[str] = None
    estimated_unit_price: Optional[float] = None
    estimated_total_price: Optional[float] = None
    currency: Optional[str] = None
    specifications: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedTenderRecord:
    """
    Enhanced unified tender record supporting:
    - Multiple data sources with varying structures
    - Lots and items hierarchies
    - Enhanced semantic data with computed insights
    - Integrated document parsing and management
    - Vector search optimization
    - Full processing pipeline tracking
    """

    # === CORE IDENTIFIERS ===
    tender_id: str  # Unique across all sources (e.g., "NEN_N006/25/V00015462")
    source_system: str  # NEN, TED, VZZ, etc.
    source_tender_id: str  # Original system ID

    id: UUID = field(init=False)

    # === RAW SOURCE DATA ===
    raw_scraped_data: Dict[str, Any]  # Original scraped data
    source_metadata: Dict[str, Any] = field(default_factory=dict)  # Source-specific metadata

    # === UNIFIED FIELDS ===
    # Basic Info
    title: Optional[str] = None
    description: Optional[str] = None
    contracting_authority: Optional[str] = None
    contracting_authority_type: Optional[str] = None  # public, private, semi-public
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_person: Optional[str] = None

    # Classification
    cpv_code: Optional[str] = None
    cpv_description: Optional[str] = None
    contract_type: Optional[str] = None  # supply, service, works, concession
    procedure_type: Optional[str] = None  # open, restricted, negotiated, etc.

    # Timeline & Status
    status: Optional[str] = None
    publication_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    opening_date: Optional[datetime] = None
    estimated_start_date: Optional[datetime] = None
    estimated_end_date: Optional[datetime] = None
    estimated_duration_days: Optional[int] = None

    # Financial (standardized to EUR)
    estimated_value_eur: Optional[float] = None
    estimated_value_original: Optional[str] = None
    currency_original: Optional[str] = None
    tender_size: Optional[str] = None  # XS, S, M, L, XL
    vat_included: Optional[bool] = None

    # Location
    location: Optional[str] = None
    location_code: Optional[str] = None
    country_code: Optional[str] = None
    nuts_code: Optional[str] = None  # EU NUTS codes

    # Framework and Structure
    is_framework: bool = False
    has_lots: bool = False
    lots: List[TenderLot] = field(default_factory=list)
    items: List[TenderItem] = field(default_factory=list)  # Direct items if no lots

    # Additional Classifications
    eligibility_criteria: List[str] = field(default_factory=list)
    required_qualifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)

    # URLs and References
    detail_url: Optional[str] = None
    documents_url: Optional[str] = None
    related_notices: List[str] = field(default_factory=list)

    # === DOCUMENT DATA ===
    document_infos: List[Dict[str, str]] = field(default_factory=list)  # Raw document info from scraping
    parsed_documents: List[ParsedDocumentData] = field(default_factory=list)  # Parsed document objects
    document_summaries: Dict[str, str] = field(default_factory=dict)  # doc_id -> summary

    # === ENHANCED SEMANTIC DATA ===
    semantic_data: Optional[Any] = None

    # === PROCESSING METADATA ===
    processing_stage: ProcessingStage = ProcessingStage.RAW_SCRAPED
    scraped_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    processing_version: str = "3.0"
    processing_errors: List[str] = field(default_factory=list)
    processing_warnings: List[str] = field(default_factory=list)

    # Data quality metrics
    data_quality_score: Optional[float] = None
    completeness_percentage: Optional[float] = None

    def __post_init__(self):
        self.id = uuid5(NAMESPACE_DNS, self.tender_id)
        if not self.scraped_at:
            self.scraped_at = datetime.now(timezone.utc).isoformat()
        if not self.processed_at:
            self.processed_at = datetime.now(timezone.utc).isoformat()

    # === PROPERTY CHECKS ===
    @property
    def is_documents_available(self) -> bool:
        """Check if tender has downloadable documents"""
        return len(self.document_infos) > 0

    @property
    def is_documents_parsed(self) -> bool:
        """Check if documents have been parsed"""
        return len(self.parsed_documents) > 0

    @property
    def has_high_quality_data(self) -> bool:
        """Check if tender has high-quality, complete data"""
        return (self.data_quality_score or 0) > 0.7

    # === UTILITY METHODS ===
    def get_all_cpv_codes(self) -> List[str]:
        """Get all CPV codes including from lots and items"""
        codes = []
        if self.cpv_code:
            codes.append(self.cpv_code)

        for lot in self.lots:
            if lot.cpv_code:
                codes.append(lot.cpv_code)

        for item in self.items:
            if item.cpv_code:
                codes.append(item.cpv_code)

        return list(set(codes))

    def get_total_estimated_value(self) -> Optional[float]:
        """Calculate total estimated value including lots"""
        if self.estimated_value_eur:
            return self.estimated_value_eur

        # Try to sum from lots
        if self.lots:
            total = 0
            for lot in self.lots:
                if lot.estimated_value_eur:
                    total += lot.estimated_value_eur
            return total if total > 0 else None

        # Try LLM-computed value
        if (self.semantic_data and
                self.semantic_data.estimated_value_eur_computed):
            return self.semantic_data.estimated_value_eur_computed

        return None

    def add_processing_error(self, error: str, stage: Optional[str] = None):
        """Add processing error with context"""
        error_msg = f"[{stage}] {error}" if stage else error
        self.processing_errors.append(error_msg)

    def add_processing_warning(self, warning: str, stage: Optional[str] = None):
        """Add processing warning with context"""
        warning_msg = f"[{stage}] {warning}" if stage else warning
        self.processing_warnings.append(warning_msg)

    def prepare_metadata_for_llm(self) -> Dict[str, Any]:
        """
        Prepare comprehensive metadata for LLM processing.
        Only includes data available BEFORE LLM processing:
        - Scraped/mapped source data
        - Parsed documents summaries
        - Raw data from source systems
        """
        # Base metadata from scraped/mapped data
        metadata = {
            "tender_id": self.tender_id,
            "source_system": self.source_system,
            "source_tender_id": self.source_tender_id,

            # Basic scraped info
            "title": self.title,
            "description": self.description,
            "contracting_authority": self.contracting_authority,
            "contracting_authority_type": self.contracting_authority_type,

            # Classification from scraping
            "cpv_code": self.cpv_code,
            "cpv_description": self.cpv_description,
            "contract_type": self.contract_type,
            "procedure_type": self.procedure_type,

            # Status and dates
            "status": self.status,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "opening_date": self.opening_date.isoformat() if self.opening_date else None,
            "estimated_start_date": self.estimated_start_date.isoformat() if self.estimated_start_date else None,
            "estimated_end_date": self.estimated_end_date.isoformat() if self.estimated_end_date else None,
            "estimated_duration_days": self.estimated_duration_days,

            # Financial from scraping
            "estimated_value_eur": self.estimated_value_eur,
            "estimated_value_original": self.estimated_value_original,
            "currency_original": self.currency_original,
            "vat_included": self.vat_included,

            # Location from scraping
            "location": self.location,
            "location_code": self.location_code,
            "country_code": self.country_code,
            "nuts_code": self.nuts_code,

            # Structure info
            "is_framework": self.is_framework,
            "has_lots": self.has_lots,
            "lots_count": len(self.lots),
            "items_count": len(self.items),

            # Pre-extracted qualifications and criteria
            "eligibility_criteria": self.eligibility_criteria,
            "required_qualifications": self.required_qualifications,
            "languages": self.languages,

            # URLs for reference
            "detail_url": self.detail_url,
            "documents_url": self.documents_url,
            "related_notices": self.related_notices,
        }

        # Add lots information (from scraping)
        if self.lots:
            metadata["lots"] = [
                {
                    "lot_id": lot.lot_id,
                    "lot_number": lot.lot_number,
                    "title": lot.title,
                    "description": lot.description,
                    "cpv_code": lot.cpv_code,
                    "cpv_description": lot.cpv_description,
                    "estimated_value_eur": lot.estimated_value_eur,
                    "estimated_value_original": lot.estimated_value_original,
                    "currency_original": lot.currency_original,
                    "location": lot.location,
                    "items_count": len(lot.items)
                }
                for lot in self.lots
            ]

        # Add items information (from scraping)
        if self.items:
            metadata["items"] = [
                {
                    "item_id": item.item_id,
                    "name": item.name,
                    "description": item.description,
                    "quantity": item.quantity,
                    "unit": item.unit,
                    "cpv_code": item.cpv_code,
                    "estimated_unit_price": item.estimated_unit_price,
                    "estimated_total_price": item.estimated_total_price,
                    "currency": item.currency
                }
                for item in self.items[:20]  # Limit to avoid token overflow
            ]

        # Add document summaries (if documents have been parsed)
        if self.document_summaries:  # TODO remove this
            metadata["document_summaries"] = self.document_summaries

        metadata["processing_context"] = {
            "processing_stage": self.processing_stage.value,
            "has_documents": len(self.document_infos) > 0,
            "documents_parsed": len(self.parsed_documents) > 0,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None,
            "data_quality_score": self.data_quality_score
        }

        return metadata

    # === NEW DOCUMENT MANAGEMENT METHODS ===
    def add_parsed_document(self, parsed_doc: ParsedDocumentData):
        """Add a parsed document to the tender"""
        self.parsed_documents.append(parsed_doc)

    def get_important_documents(self, document_ids: List[str] = None) -> List[ParsedDocumentData]:
        """Get important documents (either specified IDs or all if None)"""
        # TODO can be used with agent
        if document_ids is None:
            return self.parsed_documents

        return [doc for doc in self.parsed_documents if doc.id in document_ids]

    def _create_primary_search_text(self) -> List[str]:
        # --- PRIMARY FIELDS (highest weight, repeated 3x) ---
        primary_fields = []

        if self.title:
            primary_fields.append(f"Title: {self.title}")

        # Updated to use semantic_data.llm_extracted structure
        if self.semantic_data and self.semantic_data.llm_extracted:
            if self.semantic_data.llm_extracted.executive_summary:
                primary_fields.append(
                    f"Executive summary: {self.semantic_data.llm_extracted.executive_summary}")

            if self.semantic_data.llm_extracted.scope_and_deliverables:
                primary_fields.append(
                    f"Scope and deliverables: {self.semantic_data.llm_extracted.scope_and_deliverables}")

        return primary_fields

    def _create_secondary_fields(self) -> List[str]:
        # --- SECONDARY FIELDS (medium weight, repeated 2x) ---
        secondary_fields = []

        if self.description:
            secondary_fields.append(f"Description: {self.description}")

        # Updated to use semantic_data.llm_extracted structure
        if self.semantic_data and self.semantic_data.llm_extracted:
            if self.semantic_data.llm_extracted.key_technologies_or_skills:
                technologies = ', '.join(self.semantic_data.llm_extracted.key_technologies_or_skills)
                secondary_fields.append(f"Required technologies: {technologies}")

            if self.semantic_data.llm_extracted.target_vendor_profile:
                secondary_fields.append(
                    f"Target vendor profile: {self.semantic_data.llm_extracted.target_vendor_profile}")

            if self.semantic_data.llm_extracted.searchable_keywords:
                keywords = ', '.join(self.semantic_data.llm_extracted.searchable_keywords)
                secondary_fields.append(f"Keywords: {keywords}")

            # Add semantic tags if available
            if self.semantic_data.llm_extracted.semantic_tags:
                tags = self.semantic_data.llm_extracted.semantic_tags
                if tags.technology_stack:
                    tech_stack = ', '.join(tags.technology_stack)
                    secondary_fields.append(f"Technology stack: {tech_stack}")
                if tags.service_types:
                    services = ', '.join(tags.service_types)
                    secondary_fields.append(f"Service types: {services}")

        return secondary_fields

    def _create_tertiary_fields(self) -> List[str]:
        # --- TERTIARY FIELDS (basic weight, included once) ---
        tertiary_fields = []

        if self.contracting_authority:
            tertiary_fields.append(f"Contracting authority: {self.contracting_authority}")

        if self.contract_type:
            tertiary_fields.append(f"Contract type: {self.contract_type}")

        if self.procedure_type:
            tertiary_fields.append(f"Procedure type: {self.procedure_type}")

        if self.cpv_description:
            tertiary_fields.append(f"CPV: {self.cpv_description}")

        # Add matching profile information
        if self.semantic_data and self.semantic_data.llm_extracted and self.semantic_data.llm_extracted.matching_profile:
            profile = self.semantic_data.llm_extracted.matching_profile
            if profile.complexity_category:
                tertiary_fields.append(f"Complexity: {profile.complexity_category}")
            if profile.tender_size_category:
                tertiary_fields.append(f"Size category: {profile.tender_size_category}")

        # Add additional semantic classifications
        if self.semantic_data and self.semantic_data.llm_extracted:
            if self.semantic_data.llm_extracted.evaluation_criteria_summary:
                tertiary_fields.append(
                    f"Evaluation criteria: {self.semantic_data.llm_extracted.evaluation_criteria_summary}")

            if self.semantic_data.llm_extracted.budget_and_timeline_context:
                tertiary_fields.append(
                    f"Budget and timeline: {self.semantic_data.llm_extracted.budget_and_timeline_context}")

        return tertiary_fields

    def create_search_text(self) -> List[str]:
        # Combine all field groups with appropriate weighting
        search_text_parts = []

        # Add primary fields three times (highest weight)
        for _ in range(3):
            search_text_parts.extend(self._create_primary_search_text())

        # Add secondary fields twice (medium weight)
        for _ in range(2):
            search_text_parts.extend(self._create_secondary_fields())

        # Add tertiary fields once (basic weight)
        search_text_parts.extend(self._create_tertiary_fields())

        # Create final search text
        return "\n\n".join(filter(None, search_text_parts))

    def create_payload(self):
        # Create payload for filtering
        payload = {
            "tender_id": self.tender_id,
            "source_system": self.source_system,
            "title": self.title,
            "contracting_authority": self.contracting_authority,
            "cpv_code": self.cpv_code,
            "estimated_value_eur": self.estimated_value_eur,
            "tender_size": self.tender_size,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "location": self.location,
            "country_code": self.country_code,
            "detail_url": self.detail_url,
            "status": self.status,
            "is_framework": self.is_framework,
            "has_lots": self.has_lots
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        # Add semantic fields to payload from new structure
        if self.semantic_data:
            if self.semantic_data.llm_extracted and self.semantic_data.llm_extracted.matching_profile:
                profile = self.semantic_data.llm_extracted.matching_profile
                payload.update({
                    "complexity_category": profile.complexity_category,
                    "tender_size_category": profile.tender_size_category,
                    "preferred_company_size": profile.preferred_company_size,
                    "required_experience_level": profile.required_experience_level
                })

            # Add computed scores
            payload.update({
                "technical_complexity_score": self.semantic_data.technical_complexity_score,
                "financial_attractiveness_score": self.semantic_data.financial_attractiveness_score,
                "market_opportunity_score": self.semantic_data.market_opportunity_score,
                "competition_risk_score": self.semantic_data.competition_risk_score
            })

        return payload

    def prepare_data_for_vector_database(self):
        return {
            "id": str(self.id),
            "text": self.create_search_text(),
            "payload": self.create_payload()
        }
