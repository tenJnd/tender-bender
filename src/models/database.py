from datetime import datetime, timezone
from typing import Dict, Any, Optional

from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class UnifiedContractRaw(Base):
    """Universal model for storing raw contract data from any source"""

    __tablename__ = 'unified_contracts_raw'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Source identification
    source_system = Column(String(20), index=True, nullable=False)  # NEN, TED, VESTNIK, etc.
    source_id = Column(String(100), index=True, nullable=False)  # Original system ID

    # Unified key fields (mapped from source-specific fields)
    procurement_name = Column(String(500))
    contracting_authority = Column(String(200))
    current_status = Column(String(100))
    procedure_type = Column(String(100))
    contract_type = Column(String(100))

    # CPV classification (standardized across sources)
    cpv_code = Column(String(20), index=True)
    cpv_description = Column(String(200), index=True)

    # Location (standardized)
    location = Column(String(200), index=True)
    location_code = Column(String(20), index=True)

    # Dates (standardized)
    publication_date = Column(DateTime, index=True)
    deadline = Column(DateTime, index=True)

    # Value (standardized)
    estimated_value = Column(String(100))
    currency = Column(String(10))

    # Framework information
    is_framework = Column(Boolean, default=False)
    has_lots = Column(Boolean, default=False)

    # URLs
    detail_url = Column(String(500))

    # Full original data as JSON
    full_raw_data = Column(JSON, nullable=False)

    # Mapped/unified data as JSON
    mapped_data = Column(JSON)

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_contract_detail(cls, contract_detail) -> 'UnifiedContractRaw':
        """Create database object from any contract detail source"""
        from dataclasses import asdict

        # Get original data
        full_raw_data = asdict(contract_detail)

        # Get mapped data using field mapper
        mapped_data = contract_detail.get_mapped_fields()

        # Extract key fields from mapped data
        return cls(
            source_system=contract_detail.source_system,
            source_id=contract_detail.get_unique_id(),
            procurement_name=mapped_data.get('procurement_procedure_name'),
            contracting_authority=mapped_data.get('contracting_authority'),
            current_status=mapped_data.get('current_status_of_the_procurement_procedure'),
            procedure_type=mapped_data.get('procurement_procedure_type'),
            contract_type=mapped_data.get('type'),
            cpv_code=mapped_data.get('code_from_the_cpv_code_list'),
            cpv_description=mapped_data.get('name_from_the_cpv_code_list'),
            location=mapped_data.get('main_place_of_performance'),
            location_code=mapped_data.get('location_code'),
            publication_date=cls._parse_date(mapped_data.get('date_of_publication_on_profile')),
            deadline=cls._parse_date(mapped_data.get('deadline_for_submitting_tenders')),
            estimated_value=mapped_data.get('estimated_value_excl_vat'),
            currency=mapped_data.get('currency'),
            is_framework=cls._parse_bool(mapped_data.get('this_is_a_framework_agreement')),
            has_lots=cls._parse_bool(mapped_data.get('division_into_lots')),
            detail_url=contract_detail.detail_url,
            full_raw_data=full_raw_data,
            mapped_data=mapped_data
        )

    def get_full_contract_detail(self) -> Dict[str, Any]:
        """Return the full contract data from JSON field"""
        return self.full_raw_data

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse date strings in various formats"""
        # Implementation same as before
        pass

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        """Parse boolean values"""
        # Implementation same as before
        pass
