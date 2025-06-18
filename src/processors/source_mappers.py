import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional

from ..models.unified_tender import UnifiedTenderRecord, ProcessingStage, TenderItem
from ..scrapers.nen_scraper import NenContractDetail

logger = logging.getLogger(__name__)


class BaseSourceMapper(ABC):
    """Abstract base class for mapping source-specific data to unified model"""

    @abstractmethod
    def map_to_unified(self, source_data: Any) -> UnifiedTenderRecord:
        """Map source-specific data to unified model"""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Get source system name"""
        pass

    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime from various formats"""
        if not date_str:
            return None

        formats = [
            '%m/%d/%Y, %I:%M %p',  # 05/26/2025, 09:00 AM
            '%d/%m/%Y, %I:%M %p',  # 26/05/2025, 09:00 AM
            '%Y-%m-%d %H:%M:%S',  # 2025-05-26 09:00:00
            '%Y-%m-%d',  # 2025-05-26
            '%d.%m.%Y %H:%M',  # 26.05.2025 09:00
            '%d.%m.%Y',  # 26.05.2025
            '%d. %m. %Y %H:%M',  # 15. 05. 2025 09:30
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _estimate_value_eur(self, value_str: Optional[str], currency: Optional[str]) -> Optional[float]:
        """Convert estimated value to EUR"""
        if not value_str:
            return None

        # Extract numeric value
        numeric_str = re.sub(r'[^\d.,]', '', value_str).replace(',', '.')
        try:
            value = float(numeric_str)
        except ValueError:
            return None

        # Simple currency conversion (in real app, use exchange rate API)
        if currency == 'CZK':
            return value / 25.0  # Approximate EUR rate
        elif currency == 'USD':
            return value * 0.85  # Approximate EUR rate
        elif currency in ('EUR', '€'):
            return value

        return value  # Assume EUR if unknown

    def _determine_tender_size(self, value_eur: Optional[float]) -> Optional[str]:
        """Determine tender size based on EUR value"""
        if not value_eur:
            return None

        if value_eur < 100_000:
            return "XS"
        elif value_eur < 500_000:
            return "S"
        elif value_eur < 2_000_000:
            return "M"
        elif value_eur < 10_000_000:
            return "L"
        else:
            return "XL"

    def _parse_bool(self, value: Any) -> bool:
        """Parse boolean from various representations"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('yes', 'ano', 'true', '1', 'da')
        return False


class NenMapper(BaseSourceMapper):
    """Maps NEN contract details to unified model"""

    def get_source_name(self) -> str:
        return "NEN"

    def map_to_unified(self, nen_contract: NenContractDetail) -> UnifiedTenderRecord:
        """Map NEN contract detail to unified model"""
        from dataclasses import asdict

        # Get raw data
        raw_data = asdict(nen_contract)

        # Extract financial info
        estimated_value_eur = self._estimate_value_eur(
            nen_contract.estimated_value_excl_vat,
            nen_contract.currency
        )

        # Extract location and location code
        location = None
        location_code = None

        if nen_contract.main_place_of_performance:
            location = nen_contract.main_place_of_performance

        if nen_contract.place_of_performance:
            if isinstance(nen_contract.place_of_performance, dict):
                location_code = nen_contract.place_of_performance.get('code')
            elif isinstance(nen_contract.place_of_performance, list) and nen_contract.place_of_performance:
                first_place = nen_contract.place_of_performance[0]
                if isinstance(first_place, dict):
                    location_code = first_place.get('code')

        # Handle contact information
        contact_person = None
        if nen_contract.name or nen_contract.surname:
            contact_person = f"{nen_contract.name or ''} {nen_contract.surname or ''}".strip()

        # Extract CPV codes - handle both main and subject matter items
        cpv_codes = []
        cpv_descriptions = []

        if nen_contract.code_from_the_cpv_code_list:
            cpv_codes.append(nen_contract.code_from_the_cpv_code_list)
        if nen_contract.name_from_the_cpv_code_list:
            cpv_descriptions.append(nen_contract.name_from_the_cpv_code_list)

        # Add CPV codes from subject matter items if available
        if nen_contract.subject_matter_items:
            for item in nen_contract.subject_matter_items:
                if isinstance(item, dict):
                    if item.get('cpv_code'):
                        cpv_codes.append(item['cpv_code'])
                    if item.get('cpv_description'):
                        cpv_descriptions.append(item['cpv_description'])

        # Create items list if subject_matter_items exist
        items = []
        if nen_contract.subject_matter_items:
            for i, item in enumerate(nen_contract.subject_matter_items):
                if isinstance(item, dict):
                    tender_item = TenderItem(
                        item_id=f"item_{i}",
                        name=item.get('name', item.get('subject_matter_name')),
                        description=item.get('description'),
                        cpv_code=item.get('cpv_code'),
                        raw_data=item
                    )
                    items.append(tender_item)

        return UnifiedTenderRecord(
            # Core identifiers
            tender_id=f"NEN_{nen_contract.nen_system_number}",
            source_system=self.get_source_name(),
            source_tender_id=nen_contract.nen_system_number or "",
            raw_scraped_data=raw_data,
            source_metadata={
                "ien_system_number": nen_contract.ien_system_number,
                "contract_registration_number_in_the_vvz": nen_contract.contract_registration_number_in_the_vvz,
                "procurement_procedure_id_on_profile": nen_contract.procurement_procedure_id_on_the_contracting_authoritys_profile,
                "public_contract_regime": nen_contract.public_contract_regime
            },

            # Basic info
            title=nen_contract.procurement_procedure_name or nen_contract.subject_matter_name,
            description=nen_contract.subject_matter_description,
            contracting_authority=nen_contract.contracting_authority,
            contact_email=nen_contract.email,
            contact_phone=nen_contract.phone_1,
            contact_person=contact_person,

            # Classification
            cpv_code=cpv_codes[0] if cpv_codes else None,
            cpv_description=cpv_descriptions[0] if cpv_descriptions else None,
            contract_type=nen_contract.type,
            procedure_type=nen_contract.procurement_procedure_type,

            # Timeline & Status
            status=nen_contract.current_status_of_the_procurement_procedure,
            publication_date=self._parse_datetime(nen_contract.date_of_publication_on_profile),
            deadline=self._parse_datetime(nen_contract.deadline_for_submitting_tenders),

            # Financial
            estimated_value_eur=estimated_value_eur,
            estimated_value_original=nen_contract.estimated_value_excl_vat,
            currency_original=nen_contract.currency,
            tender_size=self._determine_tender_size(estimated_value_eur),
            vat_included=False,  # NEN shows values excluding VAT

            # Location
            location=location,
            location_code=location_code,
            country_code='CZ',  # NEN is Czech system

            # Framework and lots
            is_framework=self._parse_bool(nen_contract.this_is_a_framework_agreement),
            has_lots=self._parse_bool(nen_contract.division_into_lots),

            # Additional context
            eligibility_criteria=[
                spec for spec in [nen_contract.specifications_of_the_procurement_procedure]
                if spec
            ],

            # URLs
            detail_url=nen_contract.detail_url,

            # Documents
            document_infos=nen_contract.documents or [],

            # Items
            items=items,

            # Processing metadata
            processing_stage=ProcessingStage.RAW_SCRAPED,
            scraped_at=datetime.utcnow()
        )


class SourceMapperRegistry:
    """Registry for source mappers"""

    _mappers: Dict[str, BaseSourceMapper] = {}

    @classmethod
    def register(cls, mapper: BaseSourceMapper):
        """Register a source mapper"""
        cls._mappers[mapper.get_source_name()] = mapper
        logger.info(f"Registered source mapper for {mapper.get_source_name()}")

    @classmethod
    def get_mapper(cls, source_name: str) -> BaseSourceMapper:
        """Get mapper for source"""
        if source_name not in cls._mappers:
            raise ValueError(f"No mapper registered for source: {source_name}")
        return cls._mappers[source_name]

    @classmethod
    def map_to_unified(cls, source_name: str, source_data: Any) -> UnifiedTenderRecord:
        """Map source data to unified model"""
        mapper = cls.get_mapper(source_name)
        return mapper.map_to_unified(source_data)


# Register default mappers
SourceMapperRegistry.register(NenMapper())
# SourceMapperRegistry.register(TedMapper())
